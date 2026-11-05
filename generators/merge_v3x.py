# -*- coding: utf-8 -*-
"""合并 v3 增补卷 → data/train_v3x（v3 重训用统一仓）。

构成：train_px1（40k：pxfov1 28k + lam_v1 重放 12k，7 列 aux）全量
    + train_v3adds（23k：far1M 12k + ksub1 7k + bright 4k，7 列 aux）
    = 63k 行 → dataset.pack 打平。

血缘：pxfov1 seed=20261029 k[1..3] broad18%；lam_v1 seed=20261027；
      v3adds seed=20261030/31/32（见 data/train_v3adds/meta_*.json）。
"""
import json
import os
import shutil
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchfai.dataset import pack

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_BASE = os.path.join(ROOT, "data", "train_px1")
SRC_NEW = os.path.join(ROOT, "data", "train_v3adds")
DST = os.path.join(ROOT, "data", "train_v3x")


def shard_count(d):
    return len([f for f in os.listdir(d)
                if f.startswith("shard_") and f.endswith(".npz")])


def main():
    for s, nm in ((SRC_BASE, "train_px1"), (SRC_NEW, "train_v3adds")):
        assert os.path.isdir(s), f"缺 {s}（先跑上游生成）"
    assert shard_count(SRC_NEW) >= 87, f"v3adds 分片不全（{shard_count(SRC_NEW)}/90）"
    os.makedirs(DST, exist_ok=True)

    n = 0
    shard_i = 0
    for src in (SRC_BASE, SRC_NEW):
        shards = sorted(f for f in os.listdir(src)
                        if f.startswith("shard_") and f.endswith(".npz"))
        for fn in shards:
            shutil.copy2(os.path.join(src, fn),
                         os.path.join(DST, f"shard_{shard_i:05d}.npz"))
            with np.load(os.path.join(src, fn)) as z:
                n += len(z["params"])
            shard_i += 1
        print(f"{os.path.basename(src)}: {len(shards)} 分片拷贝完", flush=True)

    metas = {}
    for f in os.listdir(SRC_NEW):
        if f.startswith("meta_") and f.endswith(".json"):
            with open(os.path.join(SRC_NEW, f), encoding="utf-8") as fh:
                metas[f[5:-5]] = json.load(fh)
    with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(merged=True, total_rows=n, shards=shard_i,
                       base="train_px1(40k)", adds=metas,
                       created=time_str(), status="merged"),
                  f, ensure_ascii=False, indent=1)
    print(f"合计 {n} 行 / {shard_i} 分片，meta 落盘，打包…", flush=True)
    pack(DST)
    print("完成：train_v3x 可训练（--data data/train_v3x --lam-cond --px-cond）")


def time_str():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
