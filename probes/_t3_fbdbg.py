# -*- coding: utf-8 -*-
"""_t3_fbdbg.py：复刻 _fallback_joint 的 7 点 λ 网格，逐一打印
solve_auto 结果与 refine 痕迹（sol 长度 / r2d_n），找 mar 未精化的原因。"""
import os
import sys
import time
import traceback
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import fabio
import pyFAI.azimuthalIntegrator
from solver_core import solve_auto, to_canvas

img = np.asarray(fabio.open(os.path.join(ZOO, "LaB6_260210.mar3450")).data,
                 float)
img[img >= 1e9] = 0.0
img[img < 0] = 0.0
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload("Mar3450.poni")
f2d = ai.getFit2D()
gt_cx, gt_cy, gt_D = f2d["centerX"], f2d["centerY"], ai.dist * 1000
canvas2, s2, _, _ = to_canvas(img, max_side=1725)
px_m2 = float(ai.detector.pixel1) * s2
lam_A = ai.wavelength * 1e10
print(f"s={s2} px_m={px_m2*1e6:.0f}um lam_A={lam_A:.4f} "
      f"GT c=({gt_cx:.1f},{gt_cy:.1f}) D={gt_D:.2f}")

for lam2 in np.geomspace(0.95, 1.05, 7):
    lam = float(lam2 * lam_A)
    t0 = time.perf_counter()
    try:
        rr = solve_auto(canvas2, lam, px_m2, cascade=True, auto_center=True,
                        refine_2d=True)
    except Exception:
        print(f"λ={lam:.4f}: EXC {traceback.format_exc()[-200:]}")
        continue
    dt = time.perf_counter() - t0
    if rr is None:
        print(f"λ={lam:.4f}: None ({dt:.0f}s)")
        continue
    ph, sol = rr
    dc = float(np.hypot(sol[0] * s2 - gt_cx, sol[1] * s2 - gt_cy))
    dD = sol[2] * px_m2 * 1000 - gt_D
    r2d = sol[10] if len(sol) >= 11 else 0
    pre = ""
    if r2d:
        pdc = float(np.hypot(sol[7] * s2 - gt_cx, sol[8] * s2 - gt_cy))
        pre = f" pre_dc={pdc:.2f} pre_dD={sol[9]*px_m2*1000-gt_D:+.2f}"
    print(f"λ={lam:.4f}: {ph:8s} ps={sol[6]:+.3f} m={sol[3]} rms={sol[4]:.2f} "
          f"dc={dc:.2f} dD={dD:+.2f} len={len(sol)} r2d={r2d} ({dt:.0f}s){pre}",
          flush=True)
