# -*- coding: utf-8 -*-
"""_t_fwdspl.py:spline 前向 vs 自研前向对拍(2026-11-08)。

joint52:mean 种子准(116.73)但每帧 fit_tilt(spline fwd)掉到
114.98(−1.9);joint8d(自研 fwd)每帧 MAD 0.085。唯一差异=fwd。
对拍:GT poni 几何(D=116.86, rot=GT)下,两 fwd 在全画布网格点
的 tth 差分布——若系统梯度 mrad 级 → spline fwd 口径 bug。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from poni_out import make_pyfai_fwd

PX1, PX2 = 103.358e-6, 102.530e-6
SPL = os.path.join(ROOT, "data", "real", "alumina",
                   "distorsion_2x2.spline")
f_native = make_pyfai_fwd(PX1, PX2, None)
f_spl = make_pyfai_fwd(PX1, PX2, SPL)

# GT poni 几何
import re
txt = open(os.path.join(ROOT, "data", "real", "alumina",
                        "al2o3_00_max_51_frames.poni")).read()
kv = dict(re.findall(r"^(\w+):\s*([-\d.eE+]+)$", txt, re.M))
D = float(kv["Distance"])
p1, p2 = float(kv["Poni1"]), float(kv["Poni2"])
r1, r2 = float(kv["Rot1"]), float(kv["Rot2"])
print(f"GT: D={D*1e3:.3f}mm rot=({np.degrees(r1):+.4f},"
      f"{np.degrees(r2):+.4f})°")

H, W = 1043, 986
d1, d2 = np.meshgrid(np.arange(H, dtype=float),
                     np.arange(W, dtype=float), indexing="ij")
d1f, d2f = d1.ravel(), d2.ravel()

t_n = f_native(D, p1, p2, r1, r2, d1f, d2f)
t_s = f_spl(D, p1, p2, r1, r2, d1f, d2f)
dt = (t_s - t_n) * 1e3   # mrad
print(f"\ntth 差(spline−自研) mrad: 中位={np.median(dt):+.3f} "
      f"均值={dt.mean():+.3f} 极值=[{dt.min():+.2f},{dt.max():+.2f}] "
      f"P95|.|={np.percentile(np.abs(dt),95):.2f}")
print("→ " + ("spline 校正为小幅畸变(合理)" if np.percentile(np.abs(dt), 95) < 1.0
              else "⚠ 系统性大偏差(口径 bug)"))

# 反向核对:GT 环位上 spline 校正方向(半径拉伸?)
r = np.hypot((d1f + 0.5) * PX1 - p1, (d2f + 0.5) * PX2 - p2)
for lo, hi in ((0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.05)):
    msk = (t_n > lo) & (t_n < hi)
    if msk.sum() > 100:
        print(f"  tth∈[{lo},{hi})rad: Δtth 中位 {np.median(dt[msk]):+.3f}mrad"
              f"(n={msk.sum()})")
