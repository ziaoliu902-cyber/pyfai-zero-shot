# -*- coding: utf-8 -*-
"""合并 practice_patch_v1(12k 实际工况 + 8k 常规) → data/finetune_v1(20k)。

供 base_v4 微调用（train.py --init-from runs/base_v3/best.pt）。
设计同 merge_v3：分片硬链接重编号、平面打包、meta 记录血缘、三切分互斥自检。
用法：python merge_ft.py [--force]
"""
import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = [(os.path.join(ROOT, "data", "practice_patch_v1", "practice"), 12000, "practice"),
       (os.path.join(ROOT, "data", "practice_patch_v1", "regular"), 8000, "regular")]
DST = os.path.join(ROOT, "data", "finetune_v1")
CHUNK = 2048


def hardlink(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    packs = []
    for root, want, name in SRC:
        shards = sorted(f for f in os.listdir(root)
                        if f.startswith("shard_") and f.endswith(".npz"))
        lens = []
        for fn in shards:
            with np.load(os.path.join(root, fn)) as z:
                lens.append(int(len(z["params"])))
        N = sum(lens)
        assert N == want, f"{name} 数量异常: {N} != {want}"
        packs.append((root, shards, lens, name))
        print(f"  {name}: {len(shards)} 分片 {N} 张")
    N = sum(sum(p[2]) for p in packs)

    with np.load(os.path.join(packs[0][0], packs[0][1][0])) as z:
        H, W = z["imgs"].shape[1:]

    if os.path.isdir(DST):
        if not a.force:
            raise SystemExit("data/finetune_v1 已存在；确认重来请加 --force")
        for f in os.listdir(DST):
            os.remove(os.path.join(DST, f))
    os.makedirs(DST, exist_ok=True)

    k = 0
    for root, shards, lens, name in packs:
        for fn in shards:
            hardlink(os.path.join(root, fn), os.path.join(DST, f"shard_{k:05d}.npz"))
            k += 1
    print(f"[1/3] 硬链接完成 {k} 分片")

    imgs = np.lib.format.open_memmap(os.path.join(DST, "imgs.npy"), mode="w+",
                                     dtype=np.uint16, shape=(N, H, W))
    params = np.empty((N, 6), np.float32)
    aux = np.empty((N, 5), np.float32)
    i = 0
    for root, shards, lens, name in packs:
        for fn, n in zip(shards, lens):
            with np.load(os.path.join(root, fn)) as z:
                imgs[i:i + n] = z["imgs"]
                params[i:i + n] = z["params"]
                aux[i:i + n] = z["aux"]
            i += n
        imgs.flush()
    assert i == N
    np.save(os.path.join(DST, "params.npy"), params)
    np.save(os.path.join(DST, "aux.npy"), aux)
    print(f"[2/3] 平面打包完成 {N}x{H}x{W}")

    prov, counts = {}, {}
    i = 0
    for root, shards, lens, name in packs:
        m = json.load(open(os.path.join(root, "meta.json"), encoding="utf-8"))
        n = sum(lens)
        prov[f"practice_patch_v1/{name}"] = {
            "root": root, "n": n, "seed": m.get("seed"),
            "rows": f"{i}..{i+n-1}",
            "config_overrides": {kk: m["config"][kk] for kk in
                                 ("min_visible_peaks", "hump_prob",
                                  "peak_counts_range", "dist_range")},
            "note": "实际工况段" if name == "practice" else "常规防遗忘段"}
        for kk, vv in m.get("counts", {}).items():
            counts[kk] = counts.get(kk, 0) + vv
        i += n
    m0 = json.load(open(os.path.join(packs[0][0], "meta.json"), encoding="utf-8"))
    meta = {"config": m0["config"], "total": N, "shard_size": 256,
            "counts": counts, "phase_names": m0["phase_names"],
            "rejects": sum(json.load(open(os.path.join(p[0], "meta.json"),
                                         encoding="utf-8")).get("rejects", 0)
                           for p in packs),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "done",
            "split_seed": 0, "split_ratios": [0.88, 0.06, 0.06],
            "provenance": dict(prov, purpose="base_v4 微调集（v4-lite）")}
    json.dump(meta, open(os.path.join(DST, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[3/3] meta.json 完成  总耗时 {time.time()-t0:.0f}s")

    from torchfai.dataset import TorchFAIDataset
    sels = {sp: TorchFAIDataset(DST, split=sp)._sel for sp in ("train", "val", "test")}
    for sp, sel in sels.items():
        print(f"  split={sp:5s} N={len(sel)}")
    assert np.intersect1d(sels["train"], sels["val"]).size == 0
    assert np.intersect1d(sels["train"], sels["test"]).size == 0
    assert np.intersect1d(sels["val"], sels["test"]).size == 0
    auxd = np.load(os.path.join(DST, "aux.npy"), mmap_mode="r")
    prac_rows = np.concatenate([np.arange(0, 12000)])       # practice 在前
    nps = int(np.intersect1d(sels["train"], prac_rows).size)
    npr = len(sels["train"]) - nps
    print(f"自检通过：三切分互斥；train 中实际工况段 {nps} / 常规 {npr} "
          f"（{nps/len(sels['train'])*100:.0f}%:{npr/len(sels['train'])*100:.0f}%）")


if __name__ == "__main__":
    main()
