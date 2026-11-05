# -*- coding: utf-8 -*-
"""_t5_inv.py：T5 严格逆映射 + 全链验证（undistort 供外部 import，
round-trip 已验证 med 0.05px；见 _t5_roundtrip.py）。

逆映射：pyFAI 正向场 raw[i,j]=F(i+dy, j+dx)（p1=pixel1*(dY+d1c)，
源码实锤）。理想网格图 C[i,j]=F(i,j) 需解 (i′+dy(i′,j′), j′+dx)= (i,j)——
不动点迭代 2 轮。位移场正则网格求一次，散点采样 map_coordinates 采场
（scipy 1.16 已删 bispeu；pyFAI list_of_points 是伪散点会 N×N 爆炸）。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")

import pyFAI.azimuthalIntegrator
from scipy.ndimage import map_coordinates


def undistort(img, sp, n_iter=2):
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    DX = np.asarray(sp.splineFuncX(xx, yy), float)
    DY = np.asarray(sp.splineFuncY(xx, yy), float)
    src_y, src_x = yy.copy(), xx.copy()
    for _ in range(n_iter):
        ddy = map_coordinates(DY, [src_y, src_x], order=1, mode="nearest")
        ddx = map_coordinates(DX, [src_y, src_x], order=1, mode="nearest")
        src_y = yy - ddy
        src_x = xx - ddx
    return map_coordinates(img, [src_y, src_x], order=1, mode="nearest")


def _run_all():
    from zoo_run import load_img
    from solver_core import (solve_auto_lambda, to_canvas, ring_radii,
                             _solve_auto_at)
    from poni_out import sector_peaks
    from torchfai.phases import PHASE_NAMES

    for img_f, poni_f in [("Frelon2k.edf", "Frelon2k.poni"),
                          ("halfccd.edf", "halfccd.poni")]:
        img = load_img(os.path.join(ZOO, img_f))
        ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
        sp = ai.detector.spline
        f2d = ai.getFit2D()
        inv = undistort(img, sp)
        print(f"\n== {poni_f} GT c=({f2d['centerX']:.0f},{f2d['centerY']:.0f}) "
              f"D={ai.dist*1000:.2f}mm")
        for tag, cur in (("raw", img), ("inv", inv)):
            canvas, s, _, _ = to_canvas(cur)
            gcx, gcy = f2d["centerX"] / s, f2d["centerY"] / s
            rs = [rp for rp, _ in ring_radii(canvas, gcx, gcy)[0]]
            sc = []
            for r0 in [r for r in rs if 8 < r][:16]:
                pts = sector_peaks(canvas, gcx, gcy, [float(r0)],
                                   nsec=24, win=4.0, mass_rel=0.10, bump=1.5)
                if len(pts) >= 10:
                    v = np.array([p[2] for p in pts])
                    sc.append(float(v.std() / max(v.mean(), 1)))
            med = float(np.median(sc)) if sc else 9.9
            print(f"   [{tag}] 画布环数={len(rs)} 扇区散布={med*100:.2f}%")
        canvas, s, _, _ = to_canvas(inv)
        px_m = float(ai.detector.pixel1) * s
        gcx, gcy = f2d["centerX"] / s, f2d["centerY"] / s
        best = None
        for lam_try in np.geomspace(0.4, 2.2, 10):
            r = _solve_auto_at(canvas, float(lam_try), px_m, gcx, gcy,
                               PHASE_NAMES, False, ai.dist,
                               d_range=(0.5, 1.5), rms_floor=0.0,
                               cascade=True)
            if r is None:
                continue
            key = (r[1][6], r[1][3], -r[1][4])
            if best is None or key > best[0]:
                best = (key, lam_try, r)
        if best is None:
            print("   GT中心诊断解: FAIL")
        else:
            _, lam_t, (ph_fit, sol) = best
            print(f"   GT中心诊断解: {ph_fit} ps={sol[6]:.3f} m={sol[3]} "
                  f"rms={sol[4]:.2f}px λg={lam_t:.3f}")
        r = solve_auto_lambda(canvas, px_m, auto_center=True)
        if r is None:
            print("   全链 JOINTλ: FAIL")
            continue
        ph_fit, sol, lam_f = r
        dcp = float(np.hypot(sol[0] * s - f2d["centerX"],
                             sol[1] * s - f2d["centerY"]))
        dD = sol[2] * px_m * 1000 - ai.dist * 1000
        print(f"   全链 JOINTλ: {ph_fit} λ̂={lam_f:.4f} "
              f"Δcenter={dcp:.1f}px ΔD@λ̂={dD:+.2f}mm m={sol[3]}")


if __name__ == "__main__":
    _run_all()
