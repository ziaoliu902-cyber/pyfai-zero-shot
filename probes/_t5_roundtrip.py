# -*- coding: utf-8 -*-
"""_t5_roundtrip.py：合成往返测试——undistort() 的方向/实现生死判决。
syn(理想图) --正向扭曲(raw=map(syn,[yy+DY,xx+DX]))--> raw --undistort-->
rec。判据：rec 与 syn 在环带上的差异（应≈插值误差 <0.5px 级），
反方向/错实现则环错位数 px~数十 px。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(os.path.join(ROOT, "data", "real", "silx_zoo"))

import pyFAI.azimuthalIntegrator
from scipy.ndimage import map_coordinates
from _t5_inv import undistort

img_f, poni_f = "Frelon2k.edf", "Frelon2k.poni"
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
sp = ai.detector.spline
f2d = ai.getFit2D()
H, W = 2048, 2048
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
DX = np.asarray(sp.splineFuncX(xx, yy), float)
DY = np.asarray(sp.splineFuncY(xx, yy), float)

# 理想图：GT 中心 12 条锐环（高斯 σ=1.2px）
cx, cy = f2d["centerX"], f2d["centerY"]
r = np.hypot(yy - cy, xx - cx)
syn = np.zeros((H, W))
for i, r0 in enumerate(np.linspace(120, 700, 12)):
    syn += (i % 3 + 1) * np.exp(-0.5 * ((r - r0) / 1.2) ** 2)

# 正向扭曲：探测器在 (i+dy, j+dx) 采样理想场
raw_sim = map_coordinates(syn, [yy + DY, xx + DX], order=1, mode="nearest")

rec = undistort(raw_sim, sp, n_iter=2)
rec1 = undistort(raw_sim, sp, n_iter=1)
# 反方向对照（直接减，一阶逆在原始点）
rec_neg = map_coordinates(raw_sim, [yy - DY, xx - DX], order=1,
                          mode="nearest")

band = np.zeros_like(syn, bool)
for r0 in np.linspace(120, 700, 12):
    band |= np.abs(r - r0) < 4
for tag, cur in (("inv x2", rec), ("inv x1", rec1), ("neg", rec_neg)):
    err = np.abs(cur - syn)[band]
    # 环带中心线误差（峰位偏移代理）：band 内 |cur−syn| 的 90 分位
    print(f"{tag:6s}: band|Δ| med={np.median(err):.3f} p90={np.percentile(err, 90):.3f} "
          f"max={err.max():.2f}")
