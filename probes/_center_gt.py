# -*- coding: utf-8 -*-
"""不依赖 GT 的中心真值：共心约束拟合（所有环共用一个圆心）。

原理：真中心 = 让"所有环的像元到同一圆心的径向散布"最小的那个点。
每条环的半径自由（各自取均值），但圆心必须相同——中心偏 1px，各环散布
同时变大，因此这个目标函数对中心敏感，而 GT 不参与。

输出：最优中心、与 GT 的差、以及由目标函数曲率估的 1σ。
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
import pyFAI
from scipy.optimize import minimize

from solver_core import to_canvas, solve_cascade, refine_geometry_2d
from torchfai.phases import phase_qs

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]


def ring_pixels(img, c0, D_px, half=3.0):
    """按初始中心 c0 与 D 选每条环的亮像元，返回列表（每条环的 (x,y) 数组）。"""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    lam_A = LAM
    tt = 2 * np.arcsin(np.clip(lam_A * Q / (4 * np.pi), 0, 0.999))
    radii = D_px * np.tan(tt)
    r_edge = min(c0[0], c0[1], W - 1 - c0[0], H - 1 - c0[1])
    out = []
    for r0 in radii:
        if r0 < 60 or r0 > 0.9 * r_edge:
            continue
        band = np.abs(np.hypot(xx - c0[0], yy - c0[1]) - r0) < half
        iy, ix = np.nonzero(band)
        if iy.size < 300:
            continue
        v = img[iy, ix]
        base = np.percentile(v, 40)
        keep = v > base + 0.5 * (np.percentile(v, 99) - base)
        if keep.sum() < 200:
            continue
        out.append((ix[keep].astype(float), iy[keep].astype(float)))
    return out


def concentric_obj(c, rings):
    tot = 0.0
    for px, py in rings:
        d = np.hypot(px - c[0], py - c[1])
        tot += float(np.sum((d - d.mean()) ** 2))
    return tot


def fit_center(rings, c_start):
    r = minimize(concentric_obj, np.asarray(c_start, float), args=(rings,),
                 method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=1e-6, maxiter=4000))
    c = r.x
    # 曲率 → 1σ：沿 x/y 各取二阶差分
    h = 0.5
    f0 = concentric_obj(c, rings)
    n = sum(p[0].size for p in rings)
    s2 = f0 / max(n - 2, 1)                    # 残差方差
    d2x = (concentric_obj(c + [h, 0], rings) - 2 * f0
           + concentric_obj(c - [h, 0], rings)) / h ** 2
    d2y = (concentric_obj(c + [0, h], rings) - 2 * f0
           + concentric_obj(c - [0, h], rings)) / h ** 2
    sx = np.sqrt(2 * s2 / max(d2x, 1e-9)) if d2x > 0 else float("inf")
    sy = np.sqrt(2 * s2 / max(d2y, 1e-9)) if d2y > 0 else float("inf")
    return c, sx, sy, r.success


print("档   GT中心            共心最优中心         差(px)     σx,σy(px)")
for d in (150, 250, 300):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gx, gy = p.poni2 / PX, p.poni1 / PX
    canvas, s, _, _ = to_canvas(img)
    sol = solve_cascade(canvas, LAM, PX * s, Q, rms_floor=1.8)
    o = refine_geometry_2d(img, sol[0] * s, sol[1] * s, sol[2] * s, Q, LAM,
                           n_iter=3)
    c0 = (o[0], o[1]) if o else (sol[0] * s, sol[1] * s)
    rings = ring_pixels(img, c0, o[2] if o else sol[2] * s)
    c, sx, sy, ok = fit_center(rings, c0)
    dist = np.hypot(c[0] - gx, c[1] - gy)
    print(f"{d:>4} ({gx:7.1f},{gy:7.1f}) ({c[0]:8.2f},{c[1]:8.2f})"
          f" {dist:8.1f}   ({sx:.2f},{sy:.2f})  环{len(rings)}")
