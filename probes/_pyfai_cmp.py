# -*- coding: utf-8 -*-
"""对比：同一批控制点上，pyFAI 的 GeometryRefinement vs 我们的 fit_tilt。

pyFAI 的做法（标准流程）：
  1. 找环点（GUI 用 massif 连通域；这里用同样的扇区峰位，保证比较的是
     "精修引擎"而不是找点算法）；
  2. 数据 = (x, y, ring_index[, weight])；
  3. GeometryRefinement(data, calibrant="CeO2", 初始几何).refine() →
     最小化模型 2θ 与标样 2θ 的差，解 dist/poni1/poni2/rot1/rot2。
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
import pyFAI
from pyFAI.geometryRefinement import GeometryRefinement

from solver_core import to_canvas, solve_cascade, refine_geometry_2d, _r_model
from torchfai.phases import phase_qs
from poni_out import fit_tilt, sector_peaks

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]


def points_from_canvas(canvas, cx, cy, D, qs, lam, nsec=36, win=6.0):
    lam_A = lam if lam > 1e-6 else lam * 1e10   # 兼容 Å / 米
    tt = 2 * np.arcsin(np.clip(lam_A * np.asarray(qs) / (4 * np.pi),
                               0, 0.999))
    radii = D * np.tan(tt)
    H, W = canvas.shape
    vis = (radii > 40) & (radii < 0.98 * min(H, W))
    pts = sector_peaks(canvas, cx, cy, radii[vis], nsec=nsec, win=win,
                       mass_rel=0.10, bump=1.5)
    # pts: (k, chi, r, mass) → (x, y, ring_index, weight)
    out = []
    for k, ch, rp, ms in pts:
        out.append((cx + rp * np.cos(ch), cy + rp * np.sin(ch), k, ms))
    return np.asarray(out, float), vis.sum()


print("档   pyFAI: ΔD(mm)  中心Δ(px)   | 我们 fit_tilt: ΔD(mm) 中心Δ(px)")
for d in (150, 250, 300):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    gx, gy = p.poni2 / PX, p.poni1 / PX
    canvas, s, _, _ = to_canvas(img)
    pxm = PX * s
    sol = solve_cascade(canvas, LAM, pxm, Q, rms_floor=1.8)
    cx, cy, D = sol[0], sol[1], sol[2]

    data, nvis = points_from_canvas(canvas, cx, cy, D, Q, LAM)
    gr = GeometryRefinement(
        data=data, calibrant="CeO2",
        dist=D * pxm, poni1=(cy + 0.5) * pxm, poni2=(cx + 0.5) * pxm,
        pixel1=pxm, pixel2=pxm, wavelength=LAM * 1e-10,
        rot1=0.0, rot2=0.0, rot3=0.0)
    gr.refine3(fix=["wavelength"])
    try:
        dD = gr.dist * 1000 - gt
        dc = np.hypot(gr.poni2 / pxm - gx / s, gr.poni1 / pxm - gy / s) * s
    except Exception as e:
        dD, dc = float("nan"), float("nan")
        print("pyFAI refine 解析失败:", e)

    tf = fit_tilt(canvas, cx, cy, D, Q, LAM, pxm, pxm, sfull=s,
                  nsec=36, mass_rel=0.10, bump=1.5)
    if tf:
        dD2 = tf["dist"] * 1000 - gt
        dc2 = np.hypot(tf["poni2"] / PX - gx, tf["poni1"] / PX - gy)
    else:
        dD2 = dc2 = float("nan")
    print(f"{d:>4}  {dD:+8.2f} {dc:8.1f}   |  {dD2:+8.2f} {dc2:8.1f}"
          f"   (控制点 {len(data)}，可见环 {nvis})")
