# -*- coding: utf-8 -*-
"""_t1_lsq_sector.py：扇区半径 LSQ 收口测试（mar3450 s=2 + P1M 对照）。"""
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
from torchfai.phases import phase_qs
import solver_core as SC


def load(p):
    img = np.asarray(fabio.open(os.path.join(ZOO, p)).data, float)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    return img


print("===== mar3450 s=2：扇区 LSQ（从 R3 网格解出发）=====", flush=True)
img = load("LaB6_260210.mar3450")
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
    os.path.join(ZOO, "Mar3450.poni"))
canvas, s, _, _ = SC.to_canvas(img, max_side=1730)
px_m = float(ai.detector.pixel1) * s
gt_D = ai.dist * 1000.0
r = SC.solve_auto(canvas, 0.3597, px_m, cascade=True, auto_center=True)
ph, sol = r
print(f"起点: {ph} m={sol[3]} ps={sol[6]:+.3f} D={sol[2]*px_m*1000:.1f}mm "
      f"c=({sol[0]:.1f},{sol[1]:.1f})", flush=True)
for lam_frac in (0.20,):
    out = SC.refine_lambda_d(canvas, sol[0], sol[1], sol[2], 0.3597,
                             phase_qs("LaB6"), dbg=True, lam_frac=lam_frac)
    if out:
        lam_h, D_h, n, rms, sig = out
        print(f"→ λ̂={lam_h:.4f} (期望0.3705±0.5%, "
              f"偏差{abs(lam_h/0.3705-1)*100:.2f}%) "
              f"D@λ̂={D_h*px_m*1000:.2f}mm "
              f"ΔD={D_h*px_m*1000-gt_D:+.2f}mm σ={sig:.4f} n={n} "
              f"rms={rms:.3f}px", flush=True)

print("\n===== Pilatus1M 对照（应仍判简并）=====", flush=True)
img2 = load("Pilatus1M.edf")
ai2 = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
    os.path.join(ZOO, "Pilatus1M.poni"))
c2, s2_, _, _ = SC.to_canvas(img2)
px2 = float(ai2.detector.pixel1) * s2_
gt2 = ai2.dist * 1000.0
r2 = SC.solve_auto(c2, 1.0, px2, cascade=True, auto_center=True)
ph2, sol2 = r2
print(f"起点: {ph2} m={sol2[3]} D={sol2[2]*px2*1000:.1f}mm", flush=True)
out2 = SC.refine_lambda_d(c2, sol2[0], sol2[1], sol2[2], 1.0,
                          phase_qs("AgBh"), dbg=True, lam_frac=0.20)
if out2:
    lam_h, D_h, n, rms, sig = out2
    print(f"→ λ̂={lam_h:.4f} σ={sig:.4f} D@λ̂={D_h*px2*1000:.2f}mm "
          f"(GT {gt2:.1f})", flush=True)
