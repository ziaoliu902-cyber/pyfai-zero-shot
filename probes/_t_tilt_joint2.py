# -*- coding: utf-8 -*-
"""_t_tilt_joint2.py：倾角联合 λ–D 阳性对照——Eiger4M（2θmax≈40°，
1D refine_lambda_d 曾 λ̂=0.9214 σ<2% 放行）+ Eiger9M（≈30°，σ16%）。
同 _t_tilt_joint 的固定 λ 曲线 + 联合 6 参。若 Eiger4M 出内部极小
→ 方法有效，mar 的平坦是数据域界判决。"""
import os
import sys
import time
import numpy as np
from scipy.optimize import least_squares

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import fabio
import pyFAI.azimuthalIntegrator
from solver_core import solve_auto, to_canvas
from poni_out import sector_peaks, make_pyfai_fwd
from torchfai.phases import phase_qs


def scan(img_f, poni_f, lam_lo_rel=0.90, lam_hi_rel=1.10, n_scan=7):
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
    f2d = ai.getFit2D()
    gt_cx, gt_cy = f2d["centerX"], f2d["centerY"]
    gt_D, gt_lam = ai.dist * 1000, ai.wavelength * 1e10
    canvas, s, _, _ = to_canvas(img)
    px_m = float(ai.detector.pixel1) * s
    px_nat = float(ai.detector.pixel1)
    r0 = solve_auto(canvas, gt_lam, px_m, cascade=True, auto_center=True)
    ph, sol = r0
    cx, cy, D_px = sol[0], sol[1], sol[2]
    tth_max = np.degrees(np.arctan(min(canvas.shape) * px_m * 0.7 / 1
                                   / (D_px * px_m)))
    print(f"\n== {img_f} {ph} 2θmax~{tth_max:.0f}° GT λ={gt_lam:.4f} "
          f"D={gt_D:.2f} 积={gt_lam*gt_D:.1f}")
    qs = np.asarray(phase_qs(ph), float)
    lam0 = gt_lam
    tt0 = 2 * np.arcsin(np.clip(qs * lam0 / (4 * np.pi), 0, 0.999))
    radii0 = D_px * np.tan(tt0)
    pts = sector_peaks(canvas, cx, cy, radii0, nsec=36, win=6.0,
                       mass_rel=0.10, bump=1.5)
    ri = np.array([p[0] for p in pts])
    rp = np.array([p[2] for p in pts])
    ms = np.array([p[3] for p in pts])
    rows = cy + rp * np.sin(np.array([p[1] for p in pts]))
    cols = cx + rp * np.cos(np.array([p[1] for p in pts]))
    d1f = (rows + 0.5) * s - 0.5
    d2f = (cols + 0.5) * s - 0.5
    tt_t = tt0[ri]
    wgt = np.sqrt(ms); wgt = wgt / wgt.mean()
    print(f"   控制点 {len(pts)}pt/{len(set(ri))}环 r∈[{rp.min():.0f},"
          f"{rp.max():.0f}]px")
    fwd = make_pyfai_fwd(px_nat, px_nat, None)
    dist0 = D_px * px_m
    poni1_0 = (cy + 0.5) * px_m
    poni2_0 = (cx + 0.5) * px_m

    def resid6(p):
        tt_mod = fwd(p[1], p[2], p[3], p[4], p[5], d1f, d2f)
        tt_tar = 2 * np.arcsin(np.clip(qs[ri] * p[0] / (4 * np.pi), 0,
                                       0.999))
        return (tt_mod - tt_tar) * wgt

    res6 = least_squares(resid6, [lam0, dist0, poni1_0, poni2_0, 0, 0],
                         loss="soft_l1", f_scale=1.5e-3,
                         x_scale=[0.01, 0.005, 0.005, 0.005, 0.01, 0.01],
                         bounds=([lam0 * 0.85, dist0 - 0.03,
                                  poni1_0 - 40 * px_m, poni2_0 - 40 * px_m,
                                  -0.06, -0.06],
                                 [lam0 * 1.15, dist0 + 0.03,
                                  poni1_0 + 40 * px_m, poni2_0 + 40 * px_m,
                                  0.06, 0.06]))
    lam_h, dist_h = float(res6.x[0]), float(res6.x[1] * 1000)
    J = np.asarray(res6.jac); n = len(tt_t)
    s2 = max(2.0 * res6.cost / max(n - 6, 1), 1e-18)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        sig = float(np.sqrt(max(cov[0, 0], 0)) / lam_h)
    except Exception:
        sig = None
    rms6 = float(np.sqrt(np.mean((resid6(res6.x) / wgt) ** 2)))
    print(f"   [联合6参] λ̂={lam_h:.4f} σ={sig*100:.1f}% D̂={dist_h:.2f}"
          f"(Δ{dist_h-gt_D:+.2f}) 积={lam_h*dist_h:.1f} "
          f"rms={rms6*1000:.2f}mrad")
    print("   [固定λ] λ → D̂/积/rms(mrad)：")
    for lam_fix in np.linspace(lam0 * lam_lo_rel, lam0 * lam_hi_rel, n_scan):
        def resid5(p):
            tt_mod = fwd(p[0], p[1], p[2], p[3], p[4], d1f, d2f)
            tt_tar = 2 * np.arcsin(np.clip(qs[ri] * lam_fix / (4 * np.pi),
                                           0, 0.999))
            return (tt_mod - tt_tar) * wgt
        r5 = least_squares(resid5, [dist0, poni1_0, poni2_0, 0, 0],
                           loss="soft_l1", f_scale=1.5e-3,
                           x_scale=[0.005, 0.005, 0.005, 0.01, 0.01],
                           bounds=([dist0 - 0.03, poni1_0 - 40 * px_m,
                                    poni2_0 - 40 * px_m, -0.06, -0.06],
                                   [dist0 + 0.03, poni1_0 + 40 * px_m,
                                    poni2_0 + 40 * px_m, 0.06, 0.06]))
        rms5v = float(np.sqrt(np.mean((resid5(r5.x) / wgt) ** 2)))
        print(f"     λ={lam_fix:.4f}: D̂={r5.x[0]*1000:8.2f} "
              f"积={lam_fix*r5.x[0]*1000:7.1f} rms={rms5v*1000:6.2f}")


scan("Eiger4M.edf", "Eiger4M.poni")
scan("Eiger9M.h5", "Eiger9M.poni")
