# -*- coding: utf-8 -*-
"""_t1_lamref_test3.py：T1 第三轮——mar3450 s=2 联合 λ 扫描（粗网格 n_s=13
保证采进 LaB6 盆地 [0.366,0.3711]）+ LSQ 收口。

验收：λ̂ 与 0.370–0.371 差 <0.5%；D@λ̂ 与 GT 222.55 差 <2mm；相 LaB6。
"""
import os
import sys
import time

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import fabio
import pyFAI.azimuthalIntegrator
from torchfai.phases import phase_qs
from solver_core import solve_auto_lambda, to_canvas

img = np.asarray(fabio.open(os.path.join(ZOO, "LaB6_260210.mar3450")).data,
                 float)
img[img >= 1e9] = 0.0
img[img < 0] = 0.0
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
    os.path.join(ZOO, "Mar3450.poni"))
lam0 = ai.wavelength * 1e10
gt_D = ai.dist * 1000.0
canvas, s, _, _ = to_canvas(img, max_side=1730)
px_m = float(ai.detector.pixel1) * s
print(f"画布 {canvas.shape} (s={s}) 名义λ={lam0:.4f} GT D={gt_D:.2f}mm",
      flush=True)

t0 = time.perf_counter()
r = solve_auto_lambda(
    canvas, px_m, lam_lo=lam0 * 0.90, lam_hi=lam0 * 1.10, n_s=13,
    auto_center=True, refine_stages=((0.02, 7), (0.004, 7)), full=True,
    dbg=True)
dt = time.perf_counter() - t0
if r is None:
    print(f"[FAIL] 无解 ({dt:.0f}s)")
    sys.exit(1)

ph, sol, lam = r["phase"], r["sol"], r["lam"]
D = sol[2] * px_m * 1000.0
print(f"\nλ_src={r['lam_src']} λ_grid={r['lam_grid']:.4f} λ̂={lam:.4f} "
      f"σ(λ)/λ={r['lam_sig']} [{dt:.0f}s, {r['n_eval']}eval]", flush=True)
print(f"相={ph} m={sol[3]} rms={sol[4]:.2f} ps={sol[6]:+.3f}", flush=True)
print(f"D@λ̂={D:.2f}mm ΔD={D-gt_D:+.2f}mm (GT {gt_D:.2f})", flush=True)
for tp in r["top3"]:
    print(f"  粗网格top3: {tp[0]:8s} λ={tp[2]:.4f} m={tp[1][3]} "
          f"ps={tp[1][6]:+.3f} D={tp[1][2]*px_m*1000:.1f}mm", flush=True)

lam_exp = 0.3705
ok_lam = abs(lam / lam_exp - 1) < 0.005 if r["lam_src"] == "lsq" else False
ok_d = abs(D - gt_D) < 2.0
ok_ph = ph == "LaB6"
print(f"\n验收: λ̂ {'✓' if ok_lam else '✗'}(<0.5%, lsq={r['lam_src']})  "
      f"D@λ̂ {'✓' if ok_d else '✗'}(<2mm)  相 {'✓' if ok_ph else '✗'}",
      flush=True)
