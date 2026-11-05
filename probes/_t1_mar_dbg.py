# -*- coding: utf-8 -*-
"""_t1_mar_dbg.py：mar3450 s=3 画布单点复现——为什么联合 λ 扫描出 CeO2
而交接文档说 s≤3 单点 solve 出 LaB6 D=220.8。"""
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
from solver_core import solve_auto, to_canvas

img = np.asarray(fabio.open(os.path.join(ZOO, "LaB6_260210.mar3450")).data,
                 float)
img[img >= 1e9] = 0.0
img[img < 0] = 0.0
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
    os.path.join(ZOO, "Mar3450.poni"))
lam0 = ai.wavelength * 1e10
gt_D = ai.dist * 1000.0

for ms, tag in ((1150, "s=3"), (1730, "s=2"), (520, "s=7默认")):
    canvas, s, _, _ = to_canvas(img, max_side=ms)
    px_m = float(ai.detector.pixel1) * s
    for lam in (lam0, 0.3705, 0.3721):
        r1 = solve_auto(canvas, lam, px_m, cascade=True, auto_center=False)
        r2 = solve_auto(canvas, lam, px_m, cascade=True, auto_center=True)
        for ac, r in (("geo", r1), ("autoC", r2)):
            if r is None:
                print(f"{tag} λ={lam:.4f} {ac}: None")
                continue
            ph, sol = r
            print(f"{tag} λ={lam:.4f} {ac}: {ph:8s} m={sol[3]} "
                  f"rms={sol[4]:.2f} ps={sol[6]:+.3f} "
                  f"D={sol[2]*px_m*1000:7.1f}mm (GT {gt_D:.1f}, "
                  f"ΔD={sol[2]*px_m*1000-gt_D:+7.1f}) c=({sol[0]:.0f},"
                  f"{sol[1]:.0f})")
