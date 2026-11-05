# -*- coding: utf-8 -*-
"""_t5_half.py：halfccd inv 图的 λ 地形扫描（GT 中心）——找出
Al2O3 是否在 ps 地形里有真极小、在哪个 λ/D。产物：每 λ 的
top(ps,m,−rms) 及对应 phase，用于诊断 solve_auto_lambda 为何丢解。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(os.path.join(ROOT, "data", "real", "silx_zoo"))

import pyFAI.azimuthalIntegrator
from zoo_run import load_img, gt_selfcheck
from solver_core import to_canvas, _solve_auto_at
from torchfai.phases import PHASE_NAMES
from _t5_inv import undistort

img = load_img(os.path.join(ROOT, "data", "real", "silx_zoo", "halfccd.edf"))
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload("halfccd.poni")
inv = undistort(img, ai.detector.spline)
f2d = ai.getFit2D()
canvas, s, _, _ = to_canvas(inv)
px_m = float(ai.detector.pixel1) * s
gcx, gcy = f2d["centerX"] / s, f2d["centerY"] / s
print(f"canvas s={s} px_m={px_m*1e6:.1f}µm gt_c_canvas=({gcx:.1f},{gcy:.1f})")

# wide 自检拿真积：s0×λ₀(1Å)×D₀ = P_true（镜像 zoo 无 λ 分支：哑 1Å）
ai.wavelength = 1e-10
ph, n, med, ok, s0 = gt_selfcheck(inv, ai, wide=True)
print(f"inv wide 自检: {ph} {n}环 med={med*100:.2f}% s0={s0:.4f} "
      f"→ P_true={s0*ai.dist*1000:.2f}Å·mm "
      f"λ|D_gt={s0*ai.dist*1000/(ai.dist*1000):.4f}Å "
      f"(=s0)")

print("\nλ 地形（GT 中心，d_range 0.4-1.6）：")
rows = []
for lam in np.geomspace(0.15, 1.4, 17):
    r = _solve_auto_at(canvas, float(lam), px_m, gcx, gcy, PHASE_NAMES,
                       False, ai.dist, d_range=(0.4, 1.6), rms_floor=0.0,
                       cascade=True)
    if r is None:
        print(f"  λ={lam:.3f}: FAIL")
        continue
    ph_f, sol = r
    print(f"  λ={lam:.3f}: {ph_f:6s} ps={sol[6]:.3f} m={sol[3]:2d} "
          f"rms={sol[4]:5.2f}px D={sol[2]*px_m*1000:7.2f}mm")
    rows.append((float(lam), str(ph_f), float(sol[6])))
al = [x for x in rows if x[1] == "Al2O3"]
if al:
    print("\nAl2O3 出现处:", [(round(l, 3), round(p, 3)) for l, _, p in al])
else:
    print("\nAl2O3 全程未出现！")
