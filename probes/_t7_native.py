# -*- coding: utf-8 -*-
"""_t7_native.py：原生分辨率二次精化探针 v2（T3 留路）。

v2 教训：必须用 zoo_run.load_img（E9M 的 uint32 哨兵像素 4294967295
zoo 保留、我 v1 清零→解掉进错盆地 dc 92.9 vs 9.25）。
路径：画布 2D 精化解 → 原生 refine_geometry_2d → 原生 Δcenter/ΔD。
"""
import os
import sys
import time
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(os.path.join(ROOT, "data", "real", "silx_zoo"))

import pyFAI.azimuthalIntegrator
from zoo_run import load_img
from solver_core import solve_auto, to_canvas, refine_geometry_2d
from torchfai.phases import phase_qs

ITEMS = [
    ("Pilatus2M", "Pilatus2M.cbf", "Pilatus2M.poni", None),
    ("Eiger4M", "Eiger4M.edf", "Eiger4M.poni", None),
    ("Eiger4M_v2", "Eiger4M.edf", "Eiger4M_v2.poni", None),
    ("Eiger9M", "Eiger9M.h5", "Eiger9M.poni", None),
    ("mar3450", "LaB6_260210.mar3450", "Mar3450.poni", 0.3611),
]


def main():
    for tag, img_f, poni_f, lam_force in ITEMS:
        img = load_img(os.path.join(os.path.join(
            ROOT, "data", "real", "silx_zoo"), img_f))
        ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
        f2d = ai.getFit2D()
        gt_cx, gt_cy, gt_D = f2d["centerX"], f2d["centerY"], ai.dist * 1000
        lam_A = ai.wavelength * 1e10
        if lam_force:
            canvas, s, _, _ = to_canvas(img, max_side=1725)
            lam_use = lam_force
        else:
            canvas, s, _, _ = to_canvas(img)
            lam_use = lam_A
        px_m = float(ai.detector.pixel1) * s
        px_nat = float(ai.detector.pixel1)
        t0 = time.perf_counter()
        r = solve_auto(canvas, lam_use, px_m, cascade=True, auto_center=True,
                       refine_2d=True)
        ms = time.perf_counter() - t0
        if r is None:
            print(f"{tag}: SOLVE FAIL")
            continue
        ph, sol = r
        r2d = int(sol[10]) if len(sol) >= 11 else 0
        qs = np.asarray(phase_qs(ph), float)
        cxn, cyn, Dn = sol[0] * s, sol[1] * s, sol[2] * s
        t0 = time.perf_counter()
        out = refine_geometry_2d(img, cxn, cyn, Dn, qs, lam_use, n_iter=3)
        ms2 = time.perf_counter() - t0
        dc_pre = float(np.hypot(cxn - gt_cx, cyn - gt_cy))
        dD_pre = Dn * px_nat * 1000 - gt_D
        if out is None:
            print(f"{tag}: {ph} r2d_c={r2d} | 原生拒收 "
                  f"dc_pre={dc_pre:.2f}px dD_pre={dD_pre:+.2f}mm "
                  f"(canvas {ms:.0f}s + native {ms2:.0f}s)", flush=True)
            continue
        cx2, cy2, D2, nr = out
        dc_post = float(np.hypot(cx2 - gt_cx, cy2 - gt_cy))
        dD_post = D2 * px_nat * 1000 - gt_D
        print(f"{tag}: {ph} r2d_c={r2d} | 原生 {nr}环: "
              f"dc {dc_pre:.2f}→{dc_post:.2f}px | "
              f"dD {dD_pre:+.2f}→{dD_post:+.2f}mm "
              f"(canvas {ms:.0f}s + native {ms2:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
