# -*- coding: utf-8 -*-
"""_t3_dbg.py：Eiger4M refine_geometry_2d 死因取证。"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import fabio
import pyFAI.azimuthalIntegrator
from solver_core import solve_auto, to_canvas, refine_geometry_2d
from torchfai.phases import phase_qs

img = np.asarray(fabio.open(os.path.join(ZOO, "Eiger4M.edf")).data, float)
img[img >= 1e9] = 0.0
img[img < 0] = 0.0
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload("Eiger4M.poni")
f2d = ai.getFit2D()
gt_cx, gt_cy = f2d["centerX"], f2d["centerY"]
canvas, s, _, _ = to_canvas(img)
px_m = float(ai.detector.pixel1) * s
lam = ai.wavelength * 1e10

r = solve_auto(canvas, lam, px_m, cascade=True, auto_center=True)
ph, sol = r
print(f"1D: {ph} c=({sol[0]:.1f},{sol[1]:.1f}) D_px={sol[2]:.1f} "
      f"r_edge={min(sol[0], sol[1], canvas.shape[1]-1-sol[0], canvas.shape[0]-1-sol[1]):.0f}")

qs = np.asarray(phase_qs(ph), float)
radii = sol[2] * np.tan(2 * np.arcsin(np.clip(qs * lam / (4 * np.pi), 0, 0.999)))
print("预测环半径(画布px):", [f"{x:.0f}" for x in radii])

out = refine_geometry_2d(canvas, sol[0], sol[1], sol[2], qs, lam,
                         dbg=True)
print("refine →", out)
if out:
    print("dc pre="
          f"{np.hypot(sol[0]*s-gt_cx, sol[1]*s-gt_cy):.2f} post="
          f"{np.hypot(out[0]*s-gt_cx, out[1]*s-gt_cy):.2f}")
