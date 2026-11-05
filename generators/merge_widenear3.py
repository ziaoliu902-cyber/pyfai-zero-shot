# -*- coding: utf-8 -*-
"""merge_widenear3.py：train_v3x2 + widenear2 + widenear3 → data/train_v4w3。

meta.config 继承 SRC_BASE（v3x2）保归一化系一致；硬链接零拷贝。
"""
import json
import os
import shutil
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from torchfai.dataset import pack

SRC_BASE = os.path.join(ROOT, "data", "train_v3x2")
SRC_NEW = [os.path.join(ROOT, "data", "train_widenear2"),
           os.path.join(ROOT, "data", "train_widenear3")]
DST = os.path.join(ROOT, "data", "train_v4w3")


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
        print("旧 DST 已清", flush=True)
    os.makedirs(DST, exist_ok=True)

    n = 0
    shard_i = 0
    for src in [SRC_BASE] + SRC_NEW:
        shards = sorted(f for f in os.listdir(src)
                        if f.startswith("shard_") and f.endswith(".npz"))
        for fn in shards:
            dst = os.path.join(DST, f"shard_{shard_i:05d}.npz")
            try:
                os.link(os.path.join(src, fn), dst)
            except OSError:
                shutil.copy2(os.path.join(src, fn), dst)
            with np.load(os.path.join(src, fn)) as z:
                n += len(z["params"])
            shard_i += 1
        print(f"{os.path.basename(src)}: {len(shards)} 分片挂链完",
              flush=True)

    base_meta = json.load(open(os.path.join(SRC_BASE, "meta.json"),
                               encoding="utf-8"))
    meta = dict(
        merged=["train_v3x2", "train_widenear2", "train_widenear3"],
        total_rows=n,
        provenance=("v4w3：全域填缝 + 大遮挡（真实帧中心死区 225~280px "
                    "vs 仿真 ≤45px 的结构差）"),
        created=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        status="done",
        config=base_meta["config"],          # 基卷归一化系
    )
    if "phase_names" in base_meta:
        meta["phase_names"] = base_meta["phase_names"]
    with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    pack(DST)
    print(f"合并+pack 完成：{n} 行 → {DST}", flush=True)


if __name__ == "__main__":
    main()
