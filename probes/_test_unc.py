# -*- coding: utf-8 -*-
"""测试 uncertainty 模块：ceria 三档的 σ_D 与倾角 σ。"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio

from solver_core import to_canvas, solve_cascade
from torchfai.phases import phase_qs
from uncertainty import dist_sigma, tilt_with_sigma

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]

for d in (150, 250, 350):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    canvas, s, _, _ = to_canvas(img)
    pxm = PX * s
    sol = solve_cascade(canvas, LAM, pxm, Q, rms_floor=1.8)
    cx, cy, D = sol[0] * s, sol[1] * s, sol[2] * s
    sD = dist_sigma(img, cx, cy, D, Q, LAM, PX)
    tf, s1, s2 = tilt_with_sigma(canvas, sol[0], sol[1], sol[2],
                                 Q, LAM, pxm, pxm, s)
    tilt = "None" if tf is None else (round(tf["rot1"] * 1e3, 2),
                                      round(tf["rot2"] * 1e3, 2))
    print(f"d={d}: sigma_D={sD*PX*1000:.3f} mm   tilt={tilt} mrad   "
          f"sigma_rot=({s1*1e3:.2f},{s2*1e3:.2f}) mrad")

