# -*- coding: utf-8 -*-
"""判决：GT 中心 vs 我们拟合的中心，哪个让环更锐？

同一张 ceria 图，分别在两个中心提 0.25px 径向谱，比较首环的峰宽/峰高
（环锐 = 中心对；中心偏 → 环被抹宽、峰高下降）。
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
import pyFAI
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

from solver_core import to_canvas, solve_cascade, refine_geometry_2d
from torchfai.phases import phase_qs

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]


def profile(img, cx, cy, dr=0.25, rmax=900):
    H, W = img.shape
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - cy, xx - cx)
    bins = np.arange(0, rmax + dr, dr)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    i, v = idx[ok], img.ravel()[ok]
    cnt = np.bincount(i, minlength=len(bins) - 1)
    p = np.bincount(i, weights=v, minlength=len(bins) - 1) / np.maximum(cnt, 1)
    return bins[:-1] + dr / 2, p


def ring_sharp(prof_r, prof, r0, half=6.0):
    m = np.abs(prof_r - r0) < half
    seg = prof[m]
    if seg.size < 5:
        return 0.0, 0.0
    base = np.percentile(seg, 10)
    peak = seg.max() - base
    # 等效宽度 = 面积/峰高（越小越锐）
    area = np.sum(seg - base)
    width = area / max(peak, 1e-9)
    return float(peak), float(width)


for d in (150, 250):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gx, gy = p.poni2 / PX, p.poni1 / PX
    canvas, s, _, _ = to_canvas(img)
    sol = solve_cascade(canvas, LAM, PX * s, Q, rms_floor=1.8)
    o = refine_geometry_2d(img, sol[0] * s, sol[1] * s, sol[2] * s, Q, LAM,
                           n_iter=3)
    fx, fy = (o[0], o[1]) if o else (sol[0] * s, sol[1] * s)
    Dn = (o[2] if o else sol[2] * s) * PX * 1000

    # fit_tilt 的中心（交付 poni 用的就是这个）
    from poni_out import fit_tilt
    tf = fit_tilt(canvas, sol[0], sol[1], sol[2], Q, LAM, PX * s, PX * s,
                  sfull=s, nsec=36, mass_rel=0.10, bump=1.5)
    tx, ty = (tf["poni2"] / PX, tf["poni1"] / PX) if tf else (fx, fy)

    # 首环半径（用解出的 D）
    tt = 2 * np.arcsin(np.clip(LAM * Q[0] / (4 * np.pi), 0, 0.999))
    r0 = Dn / 1000 * np.tan(tt) / PX

    for tag, (cx, cy) in (("GT中心", (gx, gy)), ("r2d中心", (fx, fy)),
                          ("fit_tilt中心", (tx, ty))):
        pr, pp = profile(img, cx, cy)
        peak, width = ring_sharp(pr, pp, r0)
        print(f"d={d} {tag}: 首环峰高={peak:8.0f}  等效宽度={width:5.2f}px  "
              f"中心=({cx:.1f},{cy:.1f})")
    print()
