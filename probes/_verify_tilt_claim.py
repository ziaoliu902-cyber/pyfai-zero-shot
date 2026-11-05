# -*- coding: utf-8 -*-
"""核对另一窗口的说法：
  1) 2D 精化在 ceria 上是否被"完整环门"拦下？拦下的是哪几档、为什么？
  2) 最终几何是否由 fit_tilt 接管？它对 GT 的倾角误差到底多少（度）？
  3) "0.024°" 是"对 GT 的精度"还是"帧间重复性"？
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
import pyFAI

from solver_core import (to_canvas, solve_cascade, refine_geometry_2d,
                         _r_model, MIN_R)
from torchfai.phases import phase_qs
from poni_out import fit_tilt

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]

print("档   D_fit   完整环(r<0.9r_edge)  2D精化   Δrot1   Δrot2   (mrad/度)")
rot_err = []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    canvas, s, _, _ = to_canvas(img)
    sol = solve_cascade(canvas, LAM, PX * s, Q, rms_floor=1.8)
    cx, cy, D = sol[0], sol[1], sol[2]
    # 完整环计数（与 refine_geometry_2d 同门）
    H, W = canvas.shape
    r_edge = min(cx, cy, W - 1 - cx, H - 1 - cy)
    radii = _r_model(np.asarray(Q, float), LAM, D)
    n_full = int(((radii > MIN_R) & (radii < 0.9 * r_edge)).sum())
    out = refine_geometry_2d(img, cx * s, cy * s, D * s, Q, LAM, n_iter=3)
    ok2d = "✓" if out is not None else "拒"

    # fit_tilt（与 phase_fit 同参数）
    tf = fit_tilt(canvas, cx, cy, D, Q, LAM, PX * s, PX * s, sfull=s,
                  nsec=36, mass_rel=0.10, bump=1.5)
    if tf:
        e1 = (tf["rot1"] - p.rot1) * 1e3
        e2 = (tf["rot2"] - p.rot2) * 1e3
        rot_err.append((e1, e2))
        print(f"{d:>4} {sol[2]*PX*s*1000:8.1f} {n_full:>10} {ok2d:>8} "
              f"{e1:+8.1f} {e2:+8.1f}   ({np.degrees(e1/1e3):+.4f}°, "
              f"{np.degrees(e2/1e3):+.4f}°)")
    else:
        print(f"{d:>4} {sol[2]*PX*s*1000:8.1f} {n_full:>10} {ok2d:>8}   fit_tilt 失败")

if rot_err:
    a = np.array(rot_err)
    mag = np.hypot(a[:, 0], a[:, 1])
    print(f"\n对 GT 的倾角误差: 平均矢量 ({a[:,0].mean():+.2f},{a[:,1].mean():+.2f}) mrad")
    print(f"  每帧合矢量 |Δrot|: 平均 {mag.mean():.2f} mrad = "
          f"{np.degrees(mag.mean()/1e3):.4f}°  最大 {mag.max():.2f} mrad")
    # 帧间重复性（几何静止 → 散射=重复性）
    print(f"  帧间散射 σ(rot1)={a[:,0].std():.2f} mrad、σ(rot2)={a[:,1].std():.2f} mrad"
          f"（合计 {np.hypot(a[:,0].std(), a[:,1].std()):.2f} mrad = "
          f"{np.degrees(np.hypot(a[:,0].std(), a[:,1].std())/1e3):.4f}°）")
