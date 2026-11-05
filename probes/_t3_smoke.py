# -*- coding: utf-8 -*-
"""_t3_smoke.py：refine_geometry_2d 冒烟——Eiger4M（胜项）与 mar3450
（fallback λ̂ 路径）开关前后对比。"""
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
from solver_core import solve_auto, to_canvas


def load(path):
    img = np.asarray(fabio.open(path).data, float)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    return img


def run(img_f, poni_f, max_side=None, lam_override=None):
    img = load(os.path.join(ZOO, img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    f2d = ai.getFit2D()
    gt_cx, gt_cy, gt_D = f2d["centerX"], f2d["centerY"], ai.dist * 1000
    if max_side is None:
        canvas, s, _, _ = to_canvas(img)
    else:
        canvas, s, _, _ = to_canvas(img, max_side=max_side)
    px_m = float(ai.detector.pixel1) * s
    lam = lam_override or ai.wavelength * 1e10
    print(f"\n== {img_f} s={s} λ={lam:.4f} GT c=({gt_cx:.1f},{gt_cy:.1f}) "
          f"D={gt_D:.2f}mm")
    for r2d in (False, True):
        t0 = time.perf_counter()
        r = solve_auto(canvas, lam, px_m, cascade=True, auto_center=True,
                       refine_2d=r2d)
        dt = time.perf_counter() - t0
        if r is None:
            print(f"  refine_2d={r2d}: None ({dt:.1f}s)")
            continue
        ph, sol = r
        cx, cy, D_px = sol[0], sol[1], sol[2]
        dc = float(np.hypot(cx * s - gt_cx, cy * s - gt_cy))
        dD = D_px * px_m * 1000 - gt_D
        extra = ""
        if len(sol) >= 11:
            pcx, pcy, pD, nr = sol[7], sol[8], sol[9], sol[10]
            pdc = float(np.hypot(pcx * s - gt_cx, pcy * s - gt_cy))
            pdD = pD * px_m * 1000 - gt_D
            extra = (f" | pre: dc={pdc:.2f}px dD={pdD:+.2f}mm"
                     f" → post dc={dc:.2f}px dD={dD:+.2f}mm rings={nr}")
        print(f"  refine_2d={r2d}: {ph:8s} ps={sol[6]:+.3f} m={sol[3]} "
              f"dc={dc:.2f}px dD={dD:+.2f}mm ({dt:.1f}s){extra}")


run("Eiger4M.edf", "Eiger4M.poni")
run("Pilatus2M.cbf", "Pilatus2M.poni")
run("LaB6_260210.mar3450", "Mar3450.poni", max_side=1725,
    lam_override=0.3611)
