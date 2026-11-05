# -*- coding: utf-8 -*-
"""合并 train_v2(65k) + hard_patch_v1(12k 近距全相 + 3k AgBh远距) → data/train_v3(80k)。

设计同 merge_v2：v2 的 255 个分片硬链接原名续用；hard 两个子目录分片重编号 255 起。
平面打包顺序 = v2 行序 + near_mix + agbh_far，与分片排序严格一致。
meta 沿用 v1 config（归一化空间不变，与 base_v1/v2 同一目标空间）。

基准口径备忘（论文用；2026-10-26 实测修正，详见 probes/audit_train_v3.py）：
  - v3 训练用 train_v3 自身切分（80k, seed=0）；v3 的 test(4800) 对 v3 无泄漏
  - base_v2 对照：同卷 4800 上评估。实测 v2/v3 切分同用 seed=0，numpy 同种子
    不同长度置换 Spearman≈0.48 强相关 → v3-test 的 v2 块行仅 215/3914（5.5%）
    曾入 v2 训练集（若两切分独立才≈88%）→ v3-test 对 base_v2 亦近乎未见，
    同卷对比是干净对比；论文按实测 5.5% 如实注明，不写"主场/保守稳健"口径

用法：python merge_v3.py [--force]
"""
import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(ROOT, "data", "train_v2")
HP_NEAR = os.path.join(ROOT, "data", "hard_patch_v1", "near_mix")
HP_FAR = os.path.join(ROOT, "data", "hard_patch_v1", "agbh_far")
V3 = os.path.join(ROOT, "data", "train_v3")
CHUNK = 2048
NEAR_TOTAL, FAR_TOTAL = 12000, 3000


def hardlink(src: str, dst: str):
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    # ---- 0) 预检 ----
    v2_shards = sorted(f for f in os.listdir(V2) if f.startswith("shard_") and f.endswith(".npz"))
    near_shards = sorted(f for f in os.listdir(HP_NEAR) if f.startswith("shard_") and f.endswith(".npz"))
    far_shards = sorted(f for f in os.listdir(HP_FAR) if f.startswith("shard_") and f.endswith(".npz"))
    assert len(v2_shards) == 255, f"v2 分片数异常: {len(v2_shards)}"

    def shard_lens(root, names):
        out = []
        for fn in names:
            with np.load(os.path.join(root, fn)) as z:
                out.append(int(len(z["params"])))
        return out

    near_lens, far_lens = shard_lens(HP_NEAR, near_shards), shard_lens(HP_FAR, far_shards)
    NN, NF = sum(near_lens), sum(far_lens)
    assert NN == NEAR_TOTAL, f"near_mix 数量异常: {NN} != {NEAR_TOTAL}"
    assert NF == FAR_TOTAL, f"agbh_far 数量异常: {NF} != {FAR_TOTAL}"

    v2_imgs = np.load(os.path.join(V2, "imgs.npy"), mmap_mode="r")
    v2_params = np.load(os.path.join(V2, "params.npy"))
    v2_aux = np.load(os.path.join(V2, "aux.npy"))
    N2 = int(v2_imgs.shape[0])
    assert N2 == 65000, f"v2 样本数异常: {N2}"
    H, W = v2_imgs.shape[1], v2_imgs.shape[2]
    N = N2 + NN + NF

    if os.path.isdir(V3):
        if not a.force:
            raise SystemExit("data/train_v3 已存在；确认重来请加 --force")
        for f in os.listdir(V3):
            os.remove(os.path.join(V3, f))

    # ---- 1) 硬链接 ----
    os.makedirs(V3, exist_ok=True)
    for fn in v2_shards:
        hardlink(os.path.join(V2, fn), os.path.join(V3, fn))
    k = 255
    for fn in near_shards:
        hardlink(os.path.join(HP_NEAR, fn), os.path.join(V3, f"shard_{k:05d}.npz")); k += 1
    for fn in far_shards:
        hardlink(os.path.join(HP_FAR, fn), os.path.join(V3, f"shard_{k:05d}.npz")); k += 1
    print(f"[1/4] 硬链接完成：255+{len(near_shards)}+{len(far_shards)} 分片  {time.time()-t0:.0f}s")

    # ---- 2) 平面打包 imgs ----
    imgs = np.lib.format.open_memmap(os.path.join(V3, "imgs.npy"), mode="w+",
                                     dtype=np.uint16, shape=(N, H, W))
    for s in range(0, N2, CHUNK):
        e = min(s + CHUNK, N2)
        imgs[s:e] = v2_imgs[s:e]
        imgs.flush()
    i = N2
    for root, names, lens in ((HP_NEAR, near_shards, near_lens),
                              (HP_FAR, far_shards, far_lens)):
        for fn, n in zip(names, lens):
            with np.load(os.path.join(root, fn)) as z:
                imgs[i:i + n] = z["imgs"]
            i += n
        imgs.flush()
    assert i == N
    print(f"[2/4] imgs.npy 打包完成 {N}x{H}x{W} ({N*H*W*2/1e9:.1f}GB)  {time.time()-t0:.0f}s")

    # ---- 3) params / aux ----
    params = np.empty((N, 6), np.float32)
    aux = np.empty((N, 5), np.float32)
    params[:N2] = v2_params
    aux[:N2] = v2_aux
    i = N2
    for root, names, lens in ((HP_NEAR, near_shards, near_lens),
                              (HP_FAR, far_shards, far_lens)):
        for fn, n in zip(names, lens):
            with np.load(os.path.join(root, fn)) as z:
                params[i:i + n] = z["params"]
                aux[i:i + n] = z["aux"]
            i += n
    np.save(os.path.join(V3, "params.npy"), params)
    np.save(os.path.join(V3, "aux.npy"), aux)
    print(f"[3/4] params/aux 完成  {time.time()-t0:.0f}s")

    # ---- 4) meta ----
    def lmeta(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    m2, mn, mf = lmeta(os.path.join(V2, "meta.json")), \
                 lmeta(os.path.join(HP_NEAR, "meta.json")), \
                 lmeta(os.path.join(HP_FAR, "meta.json"))
    counts = m2.get("counts", {}).copy()
    for m in (mn, mf):
        for kk, vv in m.get("counts", {}).items():
            counts[kk] = counts.get(kk, 0) + vv
    meta = {
        "config": m2["config"],
        "total": N, "shard_size": m2["config"]["shard_size"],
        "counts": counts, "phase_names": m2["phase_names"],
        "rejects": m2.get("rejects", 0) + mn.get("rejects", 0) + mf.get("rejects", 0),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "done",
        "split_seed": 0, "split_ratios": [0.88, 0.06, 0.06],
        "provenance": {
            "train_v2": {"root": "data/train_v2", "n": N2,
                         "note": "前 65000 行(分片 0-254)，含 v1+far_patch 链"},
            "hard_patch_v1/near_mix": {"root": HP_NEAR, "n": NN, "seed": mn.get("seed"),
                                       "dist_range": mn["config"]["dist_range"],
                                       "forced_phase": mn.get("forced_phase"),
                                       "note": f"行 {N2}..{N2+NN-1}，近距全相补课"},
            "hard_patch_v1/agbh_far": {"root": HP_FAR, "n": NF, "seed": mf.get("seed"),
                                       "dist_range": mf["config"]["dist_range"],
                                       "forced_phase": mf.get("forced_phase"),
                                       "note": f"行 {N2+NN}..{N-1}，AgBh 远距补课"},
            "benchmark_note": "v3 test(4800) 对 v3 无泄漏；实测修正(2026-10-26, probes/audit_train_v3)："
                              "v3-test 中 v2块行 3914 张仅 215 张(5.5%)曾入 base_v2 训练集——"
                              "v2/v3 切分同用 seed=0，numpy 同种子不同长度置换 Spearman≈0.48 强相关，"
                              "v3-test 系统性落在 v2 val/test，对 base_v2 亦近乎未见"
                              " → 同卷对比为干净对比（原'约88%主场/保守稳健'系按切分比例的独立假设推算，与实测不符）；"
                              "论文按实测 5.5% 如实注明",
        },
    }
    with open(os.path.join(V3, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[4/4] meta.json 完成 → data/train_v3  总耗时 {time.time()-t0:.0f}s")

    # ---- 5) 自检 ----
    from torchfai.dataset import TorchFAIDataset
    hard_idx = np.arange(N2, N)
    for sp in ("train", "val", "test"):
        ds = TorchFAIDataset(V3, split=sp)
        n_hard = int(np.intersect1d(ds._sel, hard_idx).size)
        print(f"  split={sp:5s}  N={len(ds):5d}  其中补课样本={n_hard}")
    tr, va, te = (TorchFAIDataset(V3, split=s)._sel for s in ("train", "val", "test"))
    assert np.intersect1d(tr, va).size == 0 and np.intersect1d(tr, te).size == 0 \
        and np.intersect1d(va, te).size == 0, "切分重叠！"
    print("自检通过：三份互斥，补课样本三段都有分布 ✅")


if __name__ == "__main__":
    main()
