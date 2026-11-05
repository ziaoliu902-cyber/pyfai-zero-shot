# -*- coding: utf-8 -*-
"""原理化增益标定：哪个训练域统计量，匹配后自动复现经验最优 s≈4~8？

统计量候选（都只算非零像素，抗 gap/死区）：
  med_log   = median(log1p(counts))
  p99_log   = p99(log1p(counts))
  mean_log  = mean(log1p(counts))
  p999_cnt  = p99.9(counts)          ← 现行 match_gain 用的
方法：训练域(几百张)算统计量参考值 → 每帧 rebin 画布二分解 s 使统计量=参考值
     → 看各统计量隐含的 s 是否帧间一致、是否落在经验最优 4~8。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI

from real_refine_v6 import match_gain

src = open(os.path.join(ROOT, "probes", "_rebin_test.py"),
           encoding="utf-8").read()
g = {}
exec(compile(src.split("print(f\"")[0], "rebin_fn", "exec"), g)
rebin_to_canvas = g["rebin_to_canvas"]

STATS = {
    "med_log":  lambda v: float(np.median(np.log1p(v))),
    "p99_log":  lambda v: float(np.percentile(np.log1p(v), 99)),
    "mean_log": lambda v: float(np.mean(np.log1p(v))),
    "p999_cnt": lambda v: float(np.percentile(v, 99.9)),
}


def stats_of(img):
    v = img[img > 0]
    return {k: fn(v) for k, fn in STATS.items()}


# ---- 训练域参考（val 分片抽样）----
D = os.path.join(ROOT, "data", "train_v4w2")
sh = [f for f in os.listdir(D) if f.startswith("shard_") and f.endswith(".npz")]
sh.sort()
rows = []
for fn in sh[::30]:
    with np.load(os.path.join(D, fn)) as z:
        rows.append(z["imgs"][::25])
tr = np.concatenate(rows)
print(f"训练抽样 {len(tr)} 张")
tr_stats = {k: float(np.median([STATS[k](im[im > 0]) for im in tr]))
            for k in STATS}
print("训练域参考:", {k: round(v, 3) for k, v in tr_stats.items()})

# ---- 每帧 rebin 画布：解 s 使统计量=参考 ----
print(f"\n{'帧':>5s}" + "".join(f"{k+'→s':>12s}" for k in STATS))
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    reb = np.clip(match_gain(rebin_to_canvas(img, p, 2.425)), 0, None)
    row = f"{d:>5d}"
    for k in STATS:
        ref = tr_stats[k]

        def err(s):
            return stats_of(reb * s)[k] - ref
        lo, hi = 0.05, 64.0
        if err(lo) > 0:
            row += f"{'<0.05':>12s}"
            continue
        for _ in range(40):
            mid = (lo * hi) ** 0.5
            if err(mid) < 0:
                lo = mid
            else:
                hi = mid
        row += f"{(lo*hi)**0.5:>12.2f}"
    print(row)
print("\n判读：某列 s 帧间一致且≈4~8 → 该统计量=原理标定量，替换 match_gain")
