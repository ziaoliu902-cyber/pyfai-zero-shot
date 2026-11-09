# -*- coding: utf-8 -*-
"""_b12_tex_sum.py:B12① 三分片结果汇总+判决(2026-11-07)。"""
import glob
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
GT = 116.860

rows = []
for p in glob.glob(os.path.join(ROOT, "probes", "_b12_tex_rows_*.npy")):
    rows += list(np.load(p, allow_pickle=True))
print(f"汇总 {len(rows)} 帧: " + ", ".join(r[0][6:10] for r in rows))
for f, o in sorted(rows, key=lambda r: r[0]):
    a, b = o["parab"], o["splitG"]
    line = f"{f}: "
    for lbl in ("parab", "splitG"):
        d = o[lbl]
        line += (f"{lbl} Δ={d['D']-GT:+6.3f} rms={d['rms']:4.2f} "
                 f"m={d['m']:2}  " if d else f"{lbl} 无解  ")
    print(line)

print("\n== 判决 ==")
res = {}
for lbl in ("parab", "splitG"):
    ds = np.array([o[lbl]["D"] - GT for _, o in rows if o[lbl]])
    res[lbl] = np.abs(ds).mean()
    print(f"  {lbl:7} n={len(ds)} 均值Δ={ds.mean():+.3f}mm "
          f"|Δ|均值={res[lbl]:.3f}mm "
          f"MAD={np.median(np.abs(ds-np.median(ds))):.3f}mm")
gain = res["parab"] - res["splitG"]
print(f"\n  |Δ|均值增益(splitG 相对 parab):{gain:+.3f}mm")
verdict = ("开启 _SPLIT_G=True" if gain > 0.05 else
           ("劣化 → 保持关" if gain < -0.05 else "中性 → 保持关"))
print(f"  判定:{verdict}")
