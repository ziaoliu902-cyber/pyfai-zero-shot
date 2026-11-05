# -*- coding: utf-8 -*-
"""对比三种口径的中心误差（对 GT）：经典 / r2d / fit_tilt。"""
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
from poni_out import fit_tilt

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]

print("档   GT中心            经典Δ(px)   r2dΔ(px)    tfΔ(px)")
for d in (150, 250, 300):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gx, gy = p.poni2 / PX, p.poni1 / PX
    canvas, s, _, _ = to_canvas(img)
    pxm = PX * s
    sol = solve_cascade(canvas, LAM, pxm, Q, rms_floor=1.8)
    cx, cy, D = sol[0], sol[1], sol[2]
    dc = np.hypot(cx * s - gx, cy * s - gy)
    o = refine_geometry_2d(img, cx * s, cy * s, D * s, Q, LAM, n_iter=3)
    if o is not None:
        dr = np.hypot(o[0] - gx, o[1] - gy)
        scx, scy, sD = o[0] / s, o[1] / s, o[2] / s
    else:
        dr = float("nan")
        scx, scy, sD = cx, cy, D
    tf = fit_tilt(canvas, scx, scy, sD, Q, LAM, pxm, pxm, sfull=s,
                  nsec=36, mass_rel=0.10, bump=1.5)
    if tf:
        dt = np.hypot(tf["poni2"] / PX - gx, tf["poni1"] / PX - gy)
    else:
        dt = float("nan")
    print(f"{d:>4} ({gx:7.1f},{gy:7.1f}) {dc:9.1f} {dr:9.1f} {dt:9.1f}")
