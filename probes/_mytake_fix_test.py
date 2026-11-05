# -*- coding: utf-8 -*-
"""试验：两处独立修复能否救 Pilatus1M
A) 峰提取阈值改用 r>=MIN_R 区域的动态范围（不被直射束峰压死）
B) 定心改用梯度径向对称投票（部分弧可用）

只做实验，不动 solver_core；通过 monkeypatch 注入。
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
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, sobel

import zoo_run as ZR
import solver_core as SC


def rs_center(canvas, edge_frac=0.10):
    img = gaussian_filter(np.clip(canvas, 0, None), 1.0)
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    mag = np.hypot(gx, gy)
    H, W = img.shape
    thr = np.quantile(mag, 1 - edge_frac)
    ys, xs = np.nonzero(mag > max(thr, 1e-9))
    gxs, gys = gx[ys, xs], gy[ys, xs]
    n = np.hypot(gxs, gys) + 1e-12
    ux, uy = gxs / n, gys / n
    acc = np.zeros((H, W))
    for r in np.arange(6, max(H, W) * 0.5, 2.0):
        for sgn in (-1.0, 1.0):
            cx = np.round(xs + sgn * r * ux).astype(int)
            cy = np.round(ys + sgn * r * uy).astype(int)
            ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            np.add.at(acc, (cy[ok], cx[ok]), 1.0)
    acc = gaussian_filter(acc, 1.5)
    py, px = np.unravel_index(np.argmax(acc), acc.shape)
    return float(px), float(py)


def ring_radii_fixed(canvas, cx, cy, nb=SC.NB, min_r=SC.MIN_R, prom=SC.PROM):
    H, W = canvas.shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    r = np.hypot(yy - cy, xx - cx)
    rmax = float(r.max())
    idx = (r / rmax * (nb - 1)).astype(np.int32).ravel()
    v = canvas.astype(np.float64).ravel()
    cnt = np.bincount(idx, minlength=nb)
    prof = np.bincount(idx, weights=v, minlength=nb) / np.maximum(cnt, 1)
    bg = float(np.median(prof[cnt > 0]))
    bin_px = rmax / nb
    radii = (np.arange(nb) + 0.5) * bin_px
    outer = radii >= min_r
    p_ref = float(prof[outer].max()) if outer.any() else float(prof.max())
    pk, _ = find_peaks(prof, prominence=prom * p_ref, distance=12)
    out = []
    for p in pk:
        if p <= 2 or p >= nb - 3:
            continue
        if prof[p] <= bg:
            continue
        y0, y1, y2 = prof[p - 1], prof[p], prof[p + 1]
        d = 0.5 * (y0 - y2) / max(y0 - 2 * y1 + y2, 1e-12)
        rp = (p + 0.5 + np.clip(d, -1, 1)) * bin_px
        if rp < min_r:
            continue
        out.append((rp, prof[p]))
    out.sort()
    merged = []
    for rp, h in out:
        if merged and rp - merged[-1][0] < 4.5:
            rp0, h0 = merged[-1]
            merged[-1] = ((rp0 * h0 + rp * h) / (h0 + h), h0 + h)
        else:
            merged.append((rp, h))
    good = cnt > 0.15 * np.median(cnt[cnt > 0])
    return merged, prof, bin_px, good


def run(tag, img_f, poni_f):
    img = ZR.load_img(os.path.join(ZOO, img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    canvas, s, _, _ = ZR.to_canvas(img)
    f2d = ai.getFit2D()
    gt = (f2d["centerX"] / s, f2d["centerY"] / s)
    rc = rs_center(canvas)
    print(f"### {tag}: RS center=({rc[0]:.1f},{rc[1]:.1f}) GT=({gt[0]:.1f},{gt[1]:.1f}) "
          f"d={np.hypot(rc[0]-gt[0], rc[1]-gt[1]):.1f}px")

    import math
    ph_gt, n, med, ok, s0 = ZR.gt_selfcheck(img, ai)
    print(f"GT selfcheck {ph_gt} n={n} med={med*100:.2f}% s={s0:.4f}")

    peaks, prof, bp, _ = ring_radii_fixed(canvas, rc[0], rc[1])
    print("fixed peaks:", [(round(r, 1), round(h, 0)) for r, h in peaks])

    SC.center_by_sharpness = lambda canvas, step=46, dbg=False: rc
    SC._SH_CACHE.clear()
    lam = (ai.wavelength or 1e-10)
    r = SC.solve_auto(canvas, lam, ai.detector.pixel1 * s, cascade=True)
    if r is None:
        print("solve: None")
        return
    ph, sol = r
    cx, cy, D_px, m, rms, ic, ps = sol[:7]
    D = D_px * ai.detector.pixel1 * s * 1000
    print(f"solve: {ph} D={D:.2f}mm (GT {ai.dist*1000:.2f}, dD={D-ai.dist*1000:+.2f}) "
          f"m={m} rms={rms:.2f} ps={ps:+.3f} center=({cx:.1f},{cy:.1f})")


if __name__ == "__main__":
    run("Pilatus1M", "Pilatus1M.edf", "Pilatus1M.poni")

