# -*- coding: utf-8 -*-
"""_t5_half2.py：halfccd inv 的"正确答案"与全谱地形。
①ai_flat + inv 的 wide 自检 → 真相/s0=λ|D_gt；
②solve_auto_lambda dbg=True 全地形（auto_center）看 Al2O3 是否成盆；
③GT 中心 + λ 固定 s0 的直解 sanity。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(os.path.join(ROOT, "data", "real", "silx_zoo"))

import pyFAI.azimuthalIntegrator
from pyFAI.detectors import Detector
from zoo_run import load_img, gt_selfcheck
from solver_core import to_canvas, solve_auto_lambda, solve_auto
from _t5_inv import undistort

img = load_img(os.path.join(ROOT, "data", "real", "silx_zoo", "halfccd.edf"))
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload("halfccd.poni")
inv = undistort(img, ai.detector.spline)
f2d = ai.getFit2D()
ai_flat = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(
    dist=ai.dist, poni1=ai.poni1, poni2=ai.poni2, rot1=ai.rot1,
    rot2=ai.rot2, rot3=ai.rot3, wavelength=1e-10,
    detector=Detector(pixel1=ai.detector.pixel1,
                      pixel2=ai.detector.pixel2))
print("①inv+平几何 wide 自检（=真答案 λ|D_gt）：")
ph, n, med, ok, s0 = gt_selfcheck(inv, ai_flat, wide=True)
print(f"   {ph} {n}环 med={med*100:.2f}% ok={ok} s0={s0:.4f} "
      f"→ λ|D_gt={s0:.4f}Å (D=99.47mm) 2θmax={2*np.degrees(np.arcsin(0.5*s0/1.94)):.0f}°级")

canvas, s, _, _ = to_canvas(inv)
px_m = float(ai.detector.pixel1) * s
print(f"\n②全谱地形（auto_center=True, canvas s={s}）：")
r = solve_auto_lambda(canvas, px_m, auto_center=True, dbg=True)
if r is not None:
    ph_f, sol, lam_f = r
    dcp = float(np.hypot(sol[0] * s - f2d["centerX"],
                         sol[1] * s - f2d["centerY"]))
    print(f"   最优: {ph_f} λ̂={lam_f:.4f} m={sol[3]} ps={sol[6]:.3f} "
          f"D={sol[2]*px_m*1000:.1f}mm Δcenter={dcp:.1f}px")

print(f"\n③GT 中心 + λ=s0 直解：")
rr = solve_auto(canvas, float(s0), px_m, cascade=True,
                cx0=f2d["centerX"] / s, cy0=f2d["centerY"] / s,
                refine_2d=True)
if rr is None:
    print("   FAIL")
else:
    ph_f, sol = rr
    print(f"   {ph_f} ps={sol[6]:.3f} m={sol[3]} rms={sol[4]:.2f}px "
          f"D={sol[2]*px_m*1000:.2f}mm (GT 99.47)")
