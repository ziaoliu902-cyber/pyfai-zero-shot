# -*- coding: utf-8 -*-
"""合并 train_v1(50k 六相) + far_patch_v1(15k AgBh远距) → data/train_v2(65k)。

设计要点：
  - v1 的 196 个分片用硬链接零拷贝；far 的 59 个分片重编号 196..254 后硬链接
  - 平面打包：imgs.npy 前 50000 行 = v1 顺序（直接从 v1 的 imgs.npy 搬运），
    后 15000 行 = far 分片顺序 —— 与分片排序严格一致
  - meta.json 沿用 v1 的 config（归一化常数不变，保证与 base_v1 同一目标空间），
    附加 provenance 记录来源与各自 seed
  - 切分由 TorchFAIDataset 在 65k 上按 seed=0 重新洗牌（互斥由机制保证）

用法：python merge_v2.py [--force]
"""
import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "data", "train_v1")
FAR = os.path.join(ROOT, "data", "far_patch_v1")
V2 = os.path.join(ROOT, "data", "train_v2")
CHUNK = 2048  # 每块样本数（~1GB 内存）


def hardlink(src: str, dst: str):
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="train_v2 已存在时推倒重来")
    a = ap.parse_args()

    t0 = time.time()
    # ---- 0) 预检 ----
    v1_shards = sorted(f for f in os.listdir(V1) if f.startswith("shard_") and f.endswith(".npz"))
    far_shards = sorted(f for f in os.listdir(FAR) if f.startswith("shard_") and f.endswith(".npz"))
    assert len(v1_shards) == 196, f"v1 分片数异常: {len(v1_shards)}"
    assert len(far_shards) == 59, f"far 分片数异常: {len(far_shards)}"

    v1_imgs = np.load(os.path.join(V1, "imgs.npy"), mmap_mode="r")
    v1_params = np.load(os.path.join(V1, "params.npy"))
    v1_aux = np.load(os.path.join(V1, "aux.npy"))
    N1 = int(v1_imgs.shape[0])
    assert N1 == 50000 and v1_params.shape[0] == N1, "v1 打包文件数量异常"
    H, W = v1_imgs.shape[1], v1_imgs.shape[2]

    far_lens = []
    for fn in far_shards:
        with np.load(os.path.join(FAR, fn)) as z:
            far_lens.append(int(len(z["params"])))
    NF = int(sum(far_lens))
    assert NF == 15000, f"far 样本数异常: {NF}"
    N = N1 + NF

    if os.path.isdir(V2):
        if not a.force:
            raise SystemExit("data/train_v2 已存在；确认重来请加 --force")
        for f in os.listdir(V2):
            os.remove(os.path.join(V2, f))

    # ---- 1) 建目录 + 硬链接分片（v1 原名，far 重编号续接）----
    os.makedirs(V2, exist_ok=True)
    for fn in v1_shards:
        hardlink(os.path.join(V1, fn), os.path.join(V2, fn))
    for k, fn in enumerate(far_shards):
        hardlink(os.path.join(FAR, fn), os.path.join(V2, f"shard_{196 + k:05d}.npz"))
    print(f"[1/4] 硬链接完成：{len(v1_shards)}+{len(far_shards)} 分片  {time.time()-t0:.0f}s")

    # ---- 2) 平面打包 imgs（分块搬运控内存）----
    imgs = np.lib.format.open_memmap(os.path.join(V2, "imgs.npy"), mode="w+",
                                     dtype=np.uint16, shape=(N, H, W))
    for s in range(0, N1, CHUNK):
        e = min(s + CHUNK, N1)
        imgs[s:e] = v1_imgs[s:e]
        imgs.flush()
    i = N1
    for fn, n in zip(far_shards, far_lens):
        with np.load(os.path.join(FAR, fn)) as z:
            imgs[i:i + n] = z["imgs"]
        i += n
        imgs.flush()
    assert i == N
    print(f"[2/4] imgs.npy 打包完成 {N}x{H}x{W} ({N*H*W*2/1e9:.1f}GB)  {time.time()-t0:.0f}s")

    # ---- 3) params / aux ----
    params = np.empty((N, 6), np.float32)
    aux = np.empty((N, 5), np.float32)
    params[:N1] = v1_params
    aux[:N1] = v1_aux
    i = N1
    for fn, n in zip(far_shards, far_lens):
        with np.load(os.path.join(FAR, fn)) as z:
            params[i:i + n] = z["params"]
            aux[i:i + n] = z["aux"]
        i += n
    np.save(os.path.join(V2, "params.npy"), params)
    np.save(os.path.join(V2, "aux.npy"), aux)
    print(f"[3/4] params/aux 完成  {time.time()-t0:.0f}s")

    # ---- 4) meta.json（v1 config 原样 + 计数合并 + 来源链）----
    with open(os.path.join(V1, "meta.json"), encoding="utf-8") as f:
        m1 = json.load(f)
    with open(os.path.join(FAR, "meta.json"), encoding="utf-8") as f:
        mf = json.load(f)
    counts = {k: m1.get("counts", {}).get(k, 0) + mf.get("counts", {}).get(k, 0)
              for k in m1.get("phase_names", [])}
    meta = {
        "config": m1["config"],                       # 归一化单一事实来源（含 v1 全范围）
        "total": N, "shard_size": m1["config"]["shard_size"],
        "counts": counts,
        "phase_names": m1["phase_names"],
        "rejects": m1.get("rejects", 0) + mf.get("rejects", 0),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "done",
        "split_seed": 0, "split_ratios": [0.88, 0.06, 0.06],
        "provenance": {
            "train_v1": {"root": "data/train_v1", "n": N1, "seed": m1.get("seed"),
                         "dist_range": m1["config"]["dist_range"]},
            "far_patch_v1": {"root": "data/far_patch_v1", "n": NF, "seed": mf.get("seed"),
                             "forced_phase": mf.get("forced_phase"),
                             "dist_range": mf["config"]["dist_range"]},
            "note": "v1 样本位于前 50000 行(分片 0-195)，far 样本位于后 15000 行(分片 196-254)"
        },
    }
    with open(os.path.join(V2, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[4/4] meta.json 完成 → data/train_v2  总耗时 {time.time()-t0:.0f}s")

    # ---- 5) 快速自检：三份切分大小 + far 在各 split 的分布 ----
    from torchfai.dataset import TorchFAIDataset
    sel_far = {}
    for sp in ("train", "val", "test"):
        ds = TorchFAIDataset(V2, split=sp)
        n_far = int(np.intersect1d(ds._sel, np.arange(50000, 65000)).size)
        sel_far[sp] = n_far
        print(f"  split={sp:5s}  N={len(ds):5d}  其中far={n_far}")
    # 三份互斥
    from torchfai.dataset import TorchFAIDataset as _T
    tr, va, te = (_T(V2, split=s)._sel for s in ("train", "val", "test"))
    assert np.intersect1d(tr, va).size == 0 and np.intersect1d(tr, te).size == 0 \
        and np.intersect1d(va, te).size == 0, "切分重叠！"
    print(f"自检通过：三份互斥；far 分布 {sel_far}；test 中 far 占比 "
          f"{sel_far['test']/ (len(te)*1.0)*100:.1f}%")


if __name__ == "__main__":
    main()
