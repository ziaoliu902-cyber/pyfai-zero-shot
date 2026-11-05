# -*- coding: utf-8 -*-
"""torchfai 数据集：NPZ 分片 → torch Dataset + 物理归一化。

用法：
    from torchfai.dataset import TorchFAIDataset, param_to_norm, norm_to_param
    ds = TorchFAIDataset("data/train_v1", split="train")

设计：
    - 分片惰性加载：同一时刻只持有一个打开的 npz（顺序访问时每片只开一次）
    - train/val/test 按"全局样本索引 + 固定种子置换"确定性切分（与分片边界无关）
    - 图像：uint16 → log1p → float32（压动态范围，训练时再加 channel 维）
    - 标签：6 物理参数 → [-1,1] 归一化（dist 用 log10；范围全部取自 meta.json 的
      SimConfig——单一事实来源，绝不在两处硬编码）
    - Windows 注意：DataLoader 建议 num_workers=0；如用多进程须 if __name__ == "__main__" 保护
"""
from __future__ import annotations

import os
import json
import time

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------- 归一化
def _norm_consts(cfg: dict) -> dict:
    """从 SimConfig dict 提取归一化常数（含探测器物理尺寸）。"""
    from torchfai.simulator import get_detector   # 局部导入，避免硬编码探测器尺寸
    db = get_detector(cfg["detector_name"], cfg["binning"])
    Hm, Wm = db.extent
    return dict(
        dlo=float(np.log10(cfg["dist_range"][0])),
        dhi=float(np.log10(cfg["dist_range"][1])),
        Hm=float(Hm), Wm=float(Wm),
        r_lo=np.array([cfg[f"rot{i}_range"][0] for i in (1, 2, 3)], np.float64),
        r_hi=np.array([cfg[f"rot{i}_range"][1] for i in (1, 2, 3)], np.float64),
    )


def param_to_norm(p: np.ndarray, c: dict, k=None) -> np.ndarray:
    """物理参数 (...,6) → 归一化 (...,6)，六通道全部落 [-1,1] 且近似均匀。

    k: 像素尺度（标量或 (...,)），视域域扩展卷（10-29）。poni 通道除以
    k×Hm——k>1 样本的 poni 物理上可到 k×Hm，不缩放会出 [-1,1]。
    k=None/1.0 = 旧卷行为（poni/Hm），逐位一致。
    """
    p = np.asarray(p, np.float64)
    kk = 1.0 if k is None else np.asarray(k, np.float64)
    dist_n = (np.log10(p[..., 0]) - c["dlo"]) / (c["dhi"] - c["dlo"]) * 2.0 - 1.0
    p1_n = p[..., 1] / (c["Hm"] * kk) * 2.0 - 1.0
    p2_n = p[..., 2] / (c["Wm"] * kk) * 2.0 - 1.0
    rot_n = 2.0 * (p[..., 3:6] - c["r_lo"]) / (c["r_hi"] - c["r_lo"]) - 1.0
    return np.stack([dist_n, p1_n, p2_n, rot_n[..., 0], rot_n[..., 1], rot_n[..., 2]],
                    axis=-1)


def norm_to_param(y: np.ndarray, c: dict, k=None) -> np.ndarray:
    """param_to_norm 的精确逆变换（评估/解码用）。k 语义同上。"""
    y = np.asarray(y, np.float64)
    kk = 1.0 if k is None else np.asarray(k, np.float64)
    dist = 10.0 ** ((y[..., 0] + 1.0) / 2.0 * (c["dhi"] - c["dlo"]) + c["dlo"])
    p1 = (y[..., 1] + 1.0) / 2.0 * (c["Hm"] * kk)
    p2 = (y[..., 2] + 1.0) / 2.0 * (c["Wm"] * kk)
    rot = (y[..., 3:6] + 1.0) / 2.0 * (c["r_hi"] - c["r_lo"]) + c["r_lo"]
    return np.stack([dist, p1, p2, rot[..., 0], rot[..., 1], rot[..., 2]], axis=-1)


PARAM_NAMES = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")


# ---------------------------------------------------------------- 打包
def pack(root: str, progress=print) -> None:
    """把全部分片打包成单一平面 .npy（imgs.npy / params.npy / aux.npy）。

    npz(zip) 无法按行随机寻址——随机取 1 样本要读整块 128MB；打包成平面
    uint16 数组后可用 mmap 按页访问（~0.5MB/样本）。一次性成本 ≈ 复制 24GB。
    """
    shards = sorted(f for f in os.listdir(root)
                    if f.startswith("shard_") and f.endswith(".npz"))
    assert shards, f"{root} 下没有分片"
    with np.load(os.path.join(root, shards[0])) as z0:
        H, W = z0["imgs"].shape[1:]
    lens = []
    for fn in shards:
        with np.load(os.path.join(root, fn)) as z:
            lens.append(int(len(z["params"])))
    N = int(sum(lens))

    # ---- 磁盘水位守卫（10-30 事故：merge v4w3 静默预分配 57GB 写爆 D 盘）----
    import shutil as _sh
    need = (N * H * W * 2 + N * 6 * 4 + N * 8 * 4) * 1.05   # imgs+params+aux
    free = _sh.disk_usage(root).free
    if free < need:
        raise RuntimeError(
            f"磁盘空间不足，拒绝 pack：需 ~{need/1e9:.1f}GB，"
            f"{os.path.splitdrive(root)[0]} 仅剩 {free/1e9:.1f}GB"
            f"（差 {(need-free)/1e9:.1f}GB）——先清旧卷或换盘")

    fp_imgs = os.path.join(root, "imgs.npy")
    imgs = np.lib.format.open_memmap(fp_imgs, mode="w+", dtype=np.uint16,
                                     shape=(N, H, W))
    params = np.empty((N, 6), np.float32)
    with np.load(os.path.join(root, shards[0])) as z0:
        aux = np.empty((N, z0["aux"].shape[1]), np.float32)   # 5 列旧卷 / 6 列含 λ
    i = 0
    t0 = time.time()
    for k, fn in enumerate(shards):
        with np.load(os.path.join(root, fn)) as z:
            n = lens[k]
            imgs[i:i + n] = z["imgs"]
            params[i:i + n] = z["params"]
            aux[i:i + n] = z["aux"]
        i += n
        if (k + 1) % 20 == 0 or k + 1 == len(shards):
            imgs.flush()
            progress(f"pack {k+1}/{len(shards)}  {i}/{N}  {time.time()-t0:.0f}s")
    imgs.flush()
    np.save(os.path.join(root, "params.npy"), params)
    np.save(os.path.join(root, "aux.npy"), aux)
    progress(f"打包完成：N={N}  imgs.npy {N*H*W*2/1e9:.1f} GB")


# ---------------------------------------------------------------- 数据集
class TorchFAIDataset(Dataset):
    """NPZ 分片数据集。

    split ∈ {"train","val","test"}；ratios=(0.88,0.06,0.06) → 44k/3k/3k。
    切分只依赖 (root, seed, ratios)——同参数必然得到同一划分，可复现。
    """

    SPLITS = ("train", "val", "test")

    def __init__(self, root: str, split: str = "train",
                 ratios=(0.88, 0.06, 0.06), seed: int = 0,
                 mode: str = "auto"):
        assert split in self.SPLITS, f"split 必须是 {self.SPLITS}"
        self.root = root
        with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.cfg = self.meta["config"]
        self.phase_names = self.meta.get("phase_names")
        self.norm_consts = _norm_consts(self.cfg)

        # mode：auto = 有 imgs.npy 用 mmap，否则 npz；mmap 随机访问快 ~30×
        has_packed = os.path.isfile(os.path.join(root, "imgs.npy"))
        self.mode = ("mmap" if has_packed else "npz") if mode == "auto" else mode
        if self.mode == "mmap":
            assert has_packed, f"{root} 没有 imgs.npy——先运行 pack(root)"

        self.shards = sorted(f for f in os.listdir(root)
                             if f.startswith("shard_") and f.endswith(".npz"))
        assert self.shards, f"{root} 下没有分片"
        self._lens = []
        for fn in self.shards:
            with np.load(os.path.join(root, fn)) as z:
                self._lens.append(int(len(z["params"])))
        self._offsets = np.concatenate([[0], np.cumsum(self._lens)])
        total = int(self._offsets[-1])

        if self.mode == "mmap":
            self._mimgs = np.load(os.path.join(root, "imgs.npy"), mmap_mode="r")
            self._mparams = np.load(os.path.join(root, "params.npy"))
            self._maux = np.load(os.path.join(root, "aux.npy"))
            assert int(self._mimgs.shape[0]) == total, "imgs.npy 与分片总数不一致"
        else:
            self._mimgs = self._mparams = self._maux = None

        # σq 标签（v4s1 多任务解耦）：gen_sigq_labels.py 产出的图像实测
        # 等效环宽，存在则作 aux 第 8 列（index 7）附在每样本 aux 后，
        # 前 7 列语义不变（batch_lam/batch_px 读 [:,5]/[:,6] 不受影响）。
        self._sigq = None
        sig_fp = os.path.join(root, "sigq.npy")
        if self.mode == "mmap" and os.path.isfile(sig_fp):
            self._sigq = np.load(sig_fp)
            assert len(self._sigq) == total, "sigq.npy 与 imgs.npy 行数不一致"
            with open(os.path.join(root, "sigq_meta.json"), encoding="utf-8") as f:
                _sm = json.load(f)
            self._sigq_lo = float(_sm["lo"])
            self._sigq_hi = float(_sm["hi"])

        g = np.random.default_rng(seed)
        idx = g.permutation(total)
        n_tr = int(total * ratios[0])
        n_va = int(total * ratios[1])
        if split == "train":
            sel = idx[:n_tr]
        elif split == "val":
            sel = idx[n_tr:n_tr + n_va]
        else:
            sel = idx[n_tr + n_va:]
        self._sel = np.sort(sel)
        self.split_sizes = dict(train=n_tr, val=n_va, test=total - n_tr - n_va)

        self._z = None          # 当前打开的 npz（缓存一个）
        self._zi = -1

    def __len__(self):
        return len(self._sel)

    def __getitem__(self, i: int):
        g = int(self._sel[i])
        if self.mode == "mmap":
            img = np.asarray(self._mimgs[g])
            params = self._mparams[g]
            aux = self._maux[g]
        else:
            si = int(np.searchsorted(self._offsets, g, side="right") - 1)
            row = g - int(self._offsets[si])
            if si != self._zi:
                if self._z is not None:
                    self._z.close()
                self._z = np.load(os.path.join(self.root, self.shards[si]))
                self._zi = si
            img = self._z["imgs"][row]                 # uint16 (H,W)
            params = self._z["params"][row]            # float32 (6,)
            aux = self._z["aux"][row]                  # float32 (5,)
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))
        k = float(aux[6]) if aux.shape[0] >= 7 else 1.0    # px-scale 条件（10-29）
        if self._sigq is not None:
            lg = float(np.log10(max(float(self._sigq[g]), 1e-6)))
            aux = np.append(aux, (lg - self._sigq_lo)
                            / (self._sigq_hi - self._sigq_lo) * 2.0 - 1.0
                            ).astype(np.float32)
        y = torch.from_numpy(param_to_norm(params, self.norm_consts, k).astype(np.float32))
        return x, y, torch.from_numpy(aux.copy())
