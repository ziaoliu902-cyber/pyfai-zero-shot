# -*- coding: utf-8 -*-
"""精度实验：经典解 vs 经典+原生 2D 精化（refine_geometry_2d）在 ceria 七档上的 ΔD。"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
import pyFAI

from solver_core import to_canvas, solve_cascade, refine_geometry_2d
from torchfai.phases import phase_qs

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]

print(f"{'档':>5}{'GT':>9}{'经典ΔD':>9}{'2D精化ΔD':>10}{'中心Δ':>9}{'环数':>5}")
e_cl, e_r2 = [], []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    gx, gy = p.poni2 / PX, p.poni1 / PX

    canvas, s, _, _ = to_canvas(img)
    sol = solve_cascade(canvas, LAM, PX * s, Q, rms_floor=1.8)
    if sol is None:
        print(d, "classic fail")
        continue
    cx, cy, D = sol[0], sol[1], sol[2]
    e1 = D * PX * s * 1000 - gt
    out = refine_geometry_2d(img, cx * s, cy * s, D * s, Q, LAM, n_iter=3)
    if out is None:
        print(f"{d:>5}{gt:9.1f}{e1:+9.2f}{'拒收':>10}")
        e_cl.append(abs(e1))
        continue
    ncx, ncy, nD, nr = out
    e2 = nD * PX * 1000 - gt
    dc = np.hypot(ncx - gx, ncy - gy)
    e_cl.append(abs(e1))
    e_r2.append(abs(e2))
    print(f"{d:>5}{gt:9.1f}{e1:+9.2f}{e2:+10.2f}{dc:9.1f}{nr:5d}")

print(f"\n经典 |ΔD| 平均 {np.mean(e_cl):.2f} 最差 {np.max(e_cl):.2f}")
if e_r2:
    print(f"精化 |ΔD| 平均 {np.mean(e_r2):.2f} 最差 {np.max(e_r2):.2f}（{len(e_r2)}/7 档）")
