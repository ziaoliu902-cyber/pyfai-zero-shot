# -*- coding: utf-8 -*-
"""merge_widenear.py：v3x2(67k) + widenear(9k) → data/train_v4w（76k）。

v3x2 = v3x 63k + agbhfar2 4k（v3far4 的训练基底，零回归风险）
widenear = 10-30 验尸判决的填缝卷（k 连续 1.9~2.9、σq 连续填缝、
dist 0.10~0.52m、λ 0.7~1.3Å）。微调从 v3far4/last.pt 热启动。"""
import json
import os
import shutil
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchfai.dataset import pack

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_BASE = os.path.join(ROOT, "data", "train_v3x2")
SRC_NEW = os.path.join(ROOT, "data", "train_widenear")
DST = os.path.join(ROOT, "data", "train_v4w")


def shard_count(d):
    return len([f for f in os.listdir(d)
                if f.startswith("shard_") and f.endswith(".npz")])


def main():
    assert shard_count(SRC_NEW) >= 35, \
        f"widenear 分片不全（{shard_count(SRC_NEW)}/36）"
    os.makedirs(DST, exist_ok=True)
    for f in os.listdir(DST):
        os.remove(os.path.join(DST, f))

    n = 0
    shard_i = 0
    for src in (SRC_BASE, SRC_NEW):
        shards = sorted(f for f in os.listdir(src)
                        if f.startswith("shard_") and f.endswith(".npz"))
        for fn in shards:
            dst = os.path.join(DST, f"shard_{shard_i:05d}.npz")
            try:
                os.link(os.path.join(src, fn), dst)      # 硬链接：零拷贝
            except OSError:
                shutil.copy2(os.path.join(src, fn), dst)
            with np.load(os.path.join(src, fn)) as z:
                n += len(z["params"])
            shard_i += 1
        print(f"{os.path.basename(src)}: {len(shards)} 分片挂链完",
              flush=True)

    meta = dict(
        merged=["train_v3x2", "train_widenear"], total_rows=n,
        provenance=("v4far 微调卷：填缝(k连续/σq连续)+v3x 全量重放；"
                    "验尸结论=真实宽帧环宽 σq0.015~0.03 落训练域缝隙"),
        created=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        status="done",
        config=json.load(open(os.path.join(SRC_NEW, "meta.json"),
                              encoding="utf-8"))["config"],
    )
    pn = json.load(open(os.path.join(SRC_NEW, "meta.json"),
                        encoding="utf-8")).get("phase_names")
    if pn:
        meta["phase_names"] = pn
    with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    pack(DST)
    print(f"合并+pack 完成：{n} 行 → {DST}", flush=True)


if __name__ == "__main__":
    main()
