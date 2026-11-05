# -*- coding: utf-8 -*-
"""估倾角可复现性：同一帧用 nsec=18/36 两套扇区各拟合一次，差值的合矢量
当作 1σ 量级（比 LSQ 协方差更诚实的经验下界）。"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
import pyFAI

from solver_core import to_canvas, solve_cascade
from torchfai.phases import phase_qs
from poni_out import fit_tilt

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]

print("档   rot18 (mrad)        rot36 (mrad)        差(合)    差(度)")
for d in (150, 250, 400):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    canvas, s, _, _ = to_canvas(img)
    pxm = PX * s
    sol = solve_cascade(canvas, LAM, pxm, Q, rms_floor=1.8)
    cx, cy, D = sol[0], sol[1], sol[2]
    a = fit_tilt(canvas, cx, cy, D, Q, LAM, pxm, pxm, sfull=s,
                 nsec=18, mass_rel=0.10, bump=1.5)
    b = fit_tilt(canvas, cx, cy, D, Q, LAM, pxm, pxm, sfull=s,
                 nsec=36, mass_rel=0.10, bump=1.5)
    if not (a and b):
        print(f"{d:>4}  fit_tilt 失败/拒")
        continue
    d1 = (a["rot1"] - b["rot1"]) * 1e3
    d2 = (a["rot2"] - b["rot2"]) * 1e3
    mag = np.hypot(d1, d2)
    print(f"{d:>4}  ({a['rot1']*1e3:+.2f},{a['rot2']*1e3:+.2f})   "
          f"({b['rot1']*1e3:+.2f},{b['rot2']*1e3:+.2f})   "
          f"{mag:5.2f} mrad   {np.degrees(mag/1e3):.4f}°")
