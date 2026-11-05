# -*- coding: utf-8 -*-
"""_t5_diag.py：T5 诊断——①spline/图像 binning 一致性（halfccd 像素
48.4µm vs FReLoN 原生 47µm→疑 2x2 bin，位移场需除 bin）；②画布尺度
环质量（GT 中心：ring_radii 环数 + 扇区散布，raw vs plus/2 vs plus）；
③GT 中心诊断解（隔离"定心被斑点带偏" vs "根本解不出"）。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import pyFAI.azimuthalIntegrator
from scipy.ndimage import map_coordinates
from zoo_run import load_img
from solver_core import (solve_auto_lambda, solve_auto, to_canvas,
                         ring_radii, _solve_auto_at)
from poni_out import sector_peaks
from torchfai.phases import PHASE_NAMES

CASES = [("Frelon2k.edf", "Frelon2k.poni"),
         ("halfccd.edf", "halfccd.poni")]

for img_f, poni_f in CASES:
    img = load_img(os.path.join(ZOO, img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    sp = ai.detector.spline
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx = np.asarray(sp.splineFuncX(xx, yy), float)
    dy = np.asarray(sp.splineFuncY(xx, yy), float)
    f2d = ai.getFit2D()
    print(f"\n== {poni_f} img={img.shape} px1={ai.detector.pixel1*1e6:.2f}µm")
    print(f"   spline: size={sp.getDetectorSize()} px={sp.getPixelSize()} "
          f"| img/spline bin 因子: x={sp.getDetectorSize()[0]/W:.3f} "
          f"y={sp.getDetectorSize()[1]/H:.3f}")
    variants = {
        "raw": img,
        "plus": map_coordinates(img, [yy + dy, xx + dx], order=1,
                                mode="nearest"),
    }
    # binning 假设：spline 网格是图像的 k 倍 → 位移除 k
    kx = sp.getDetectorSize()[0] / W
    ky = sp.getDetectorSize()[1] / H
    if abs(kx - 1) > 0.05 or abs(ky - 1) > 0.05:
        variants["plus/bink"] = map_coordinates(
            img, [yy + dy / ky, xx + dx / kx], order=1, mode="nearest")
    for tag, cur in variants.items():
        canvas, s, _, _ = to_canvas(cur)
        gcx, gcy = f2d["centerX"] / s, f2d["centerY"] / s
        rs = [rp for rp, _ in ring_radii(canvas, gcx, gcy)[0]]
        rs = [r for r in rs if 8 < r < 0.45 * min(canvas.shape)]
        sc = []
        for r0 in rs[:14]:
            pts = sector_peaks(canvas, gcx, gcy, [float(r0)],
                               nsec=24, win=4.0, mass_rel=0.10, bump=1.5)
            if len(pts) >= 10:
                v = np.array([p[2] for p in pts])
                sc.append(float(v.std() / max(v.mean(), 1)))
        med = float(np.median(sc)) if sc else 9.9
        print(f"   [{tag:9s}] 画布环数={len(rs)} 扇区散布={med*100:.2f}%")
    # ---- GT 中心诊断解（无 λ：粗网格 λ 扫描 × GT 中心，隔离定心因素）----
    cur = variants.get("plus/bink", variants["plus"])
    canvas, s, _, _ = to_canvas(cur)
    px_m = float(ai.detector.pixel1) * s
    gcx, gcy = f2d["centerX"] / s, f2d["centerY"] / s
    best = None
    for lam_try in np.geomspace(0.5, 2.5, 9):
        r = _solve_auto_at(canvas, float(lam_try), px_m, gcx, gcy,
                           PHASE_NAMES, False, ai.dist,
                           d_range=(0.45, 1.55), rms_floor=0.0,
                           cascade=True)
        if r is None:
            continue
        key = (r[1][6], r[1][3], -r[1][4])
        if best is None or key > best[0]:
            best = (key, lam_try, r)
    if best is None:
        print("   GT中心诊断解: FAIL")
        continue
    _, lam_t, (ph_fit, sol) = best
    print(f"   GT中心诊断解: {ph_fit} ps={sol[6]:.3f} m={sol[3]} "
          f"rms={sol[4]:.2f}px λgrid={lam_t:.3f} "
          f"Δcenter={np.hypot((sol[0]-gcx)*s, (sol[1]-gcy)*s):.1f}px "
          f"(GT={f2d['centerX']:.0f},{f2d['centerY']:.0f})")
