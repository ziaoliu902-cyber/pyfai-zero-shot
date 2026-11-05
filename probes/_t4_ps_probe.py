# -*- coding: utf-8 -*-
"""_t4_ps_probe.py：T4 前置——各 has-λ 项默认画布解的 pscore，
确认 fallback 触发阈 0.55 只打 mar3450。"""
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

CASES = [("Pilatus1M.edf", "Pilatus1M.poni", 1.0),
         ("Pilatus2M.cbf", "Pilatus2M.poni", None),
         ("Eiger4M.edf", "Eiger4M.poni", None),
         ("Eiger9M.h5", "Eiger9M.poni", None),
         ("LaB6_260210.mar3450", "Mar3450.poni", None)]

for img_f, poni_f, _ in CASES:
    if img_f.endswith(".h5"):
        import h5py
        with h5py.File(os.path.join(ZOO, img_f), "r") as f:
            hit = []

            def _w(name, obj):
                if not hit and hasattr(obj, "ndim") and obj.ndim >= 2 \
                        and name.split("/")[-1] == "data":
                    hit.append(obj)
            f.visititems(_w)
            arr = hit[0][()]
            img = np.asarray(arr[0] if arr.ndim == 3 else arr, float)
    else:
        img = np.asarray(fabio.open(os.path.join(ZOO, img_f)).data, float)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    canvas, s, _, _ = to_canvas(img)
    px_m = float(ai.detector.pixel1) * s
    lam = ai.wavelength * 1e10
    r = solve_auto(canvas, lam, px_m, cascade=True, auto_center=True)
    if r is None:
        print(f"{img_f:24s} s={s}: None", flush=True)
        continue
    ph, sol = r
    print(f"{img_f:24s} s={s}: {ph:8s} m={sol[3]} ps={sol[6]:+.3f} "
          f"D={sol[2]*px_m*1000:8.1f}mm", flush=True)
