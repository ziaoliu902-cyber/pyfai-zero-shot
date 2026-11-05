# -*- coding: utf-8 -*-
"""不依赖 GT 的对比：同一张图，pyFAI 精修中心 vs 我们中心，谁的环更"圆"。

ring_rms = 各环 Kasa 圆拟合残差中位（px）。中心对 → 环圆 → rms 小。
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
import pyFAI
from pyFAI.geometryRefinement import GeometryRefinement

from solver_core import to_canvas, solve_cascade, refine_geometry_2d
from torchfai.phases import phase_qs
from poni_out import fit_tilt, sector_peaks
from uncertainty import ring_rms

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]


def pts(canvas, cx, cy, D, nsec=36, win=6.0):
    lam_A = LAM
    tt = 2 * np.arcsin(np.clip(lam_A * np.asarray(Q) / (4 * np.pi), 0, 0.999))
    radii = D * np.tan(tt)
    H, W = canvas.shape
    vis = (radii > 40) & (radii < 0.98 * min(H, W))
    p = sector_peaks(canvas, cx, cy, radii[vis], nsec=nsec, win=win,
                     mass_rel=0.10, bump=1.5)
    return np.asarray([(cx + r * np.cos(ch), cy + r * np.sin(ch), k, m)
                       for k, ch, r, m in p], float)


print("档   pyFAI中心 rms   我们(tf) rms   r2d rms    GT中心 rms")
for d in (150, 250, 300):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gx, gy = p.poni2 / PX, p.poni1 / PX
    canvas, s, _, _ = to_canvas(img)
    pxm = PX * s
    sol = solve_cascade(canvas, LAM, pxm, Q, rms_floor=1.8)
    cx, cy, D = sol[0], sol[1], sol[2]

    data = pts(canvas, cx, cy, D)
    gr = GeometryRefinement(data=data, calibrant="CeO2",
                            dist=D * pxm, poni1=(cy + .5) * pxm,
                            poni2=(cx + .5) * pxm, pixel1=pxm, pixel2=pxm,
                            wavelength=LAM * 1e-10,
                            rot1=0., rot2=0., rot3=0.)
    gr.refine3(fix=["wavelength"])
    gcx, gcy = gr.poni2 / pxm, gr.poni1 / pxm      # 画布 px
    # 换回原生 px（D 也用画布 D 换算，保持一致）
    dpx = D * pxm / PX

    tf = fit_tilt(canvas, cx, cy, D, Q, LAM, pxm, pxm, sfull=s,
                  nsec=36, mass_rel=0.10, bump=1.5)
    tcx, tcy = (tf["poni2"] / PX, tf["poni1"] / PX) if tf else (cx * s, cy * s)
    o = refine_geometry_2d(img, cx * s, cy * s, D * s, Q, LAM, n_iter=3)
    rcx, rcy = (o[0], o[1]) if o else (cx * s, cy * s)

    a, _ = ring_rms(img, gcx * s, gcy * s, dpx, Q, LAM)
    b, _ = ring_rms(img, tcx, tcy, dpx, Q, LAM)
    c, _ = ring_rms(img, rcx, rcy, dpx, Q, LAM)
    e, _ = ring_rms(img, gx, gy, dpx, Q, LAM)
    print(f"{d:>4} {a:12.2f}px {b:12.2f}px {c:10.2f}px {e:10.2f}px")
