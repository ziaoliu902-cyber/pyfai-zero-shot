# -*- coding: utf-8 -*-
"""mar3450 亚像素定心：迭代圆拟合（径向脊线 → 逐环 Kasa → 中位聚合）。

在 1/2 分辨率（1725²）上做，环间距 17-33px，够分离。
每轮：
  1. 以当前中心提径向谱（0.5px bin），找强环；
  2. 每条环在 ±2.5px 带内取高于背景的像素，Kasa 代数圆拟合，
     2.5σ 剪枝两遍；
  3. 按拟合残差 σ 加权，取各环中心的加权中位；
  4. 阻尼更新（0.7）。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")

import fabio
import pyFAI.azimuthalIntegrator
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

import solver_core as SC


def load_pool(s=2):
    img = np.asarray(fabio.open(os.path.join(ZOO, "LaB6_260210.mar3450")).data,
                     float)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    H, W = img.shape
    h, w = H // s, W // s
    return img[:h * s, :w * s].reshape(h, s, w, s).mean(axis=(1, 3))


def radial(img, cx, cy, dr=0.5):
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - cy, xx - cx)
    rmax = float(r.max())
    bins = np.arange(0, rmax + dr, dr)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    i = idx[ok]
    v = img.ravel()[ok]
    cnt = np.bincount(i, minlength=len(bins) - 1)
    prof = np.bincount(i, weights=v, minlength=len(bins) - 1) / np.maximum(cnt, 1)
    return bins[:-1] + dr / 2, prof


def kasa(px, py):
    A = np.stack([px, py, np.ones_like(px)], 1)
    b = px ** 2 + py ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = sol
    cx, cy = a / 2, bb / 2
    r2 = c + (a * a + bb * bb) / 4
    if r2 <= 0:
        return None
    return cx, cy, np.sqrt(r2)


def refine(img, c0, n_iter=8, damp=0.7, dbg=True):
    cx, cy = c0
    yy, xx = np.indices(img.shape)
    for it in range(n_iter):
        r_ax, prof = radial(img, cx, cy, dr=0.5)
        bg = median_filter(prof, 41)
        d = prof - bg
        pk, _ = find_peaks(d, prominence=0.015 * d.max(), distance=8)
        ring_r = [(r_ax[p], d[p]) for p in pk if r_ax[p] > 120]
        ring_r.sort(key=lambda t: -t[1])
        fits = []
        for rp, h in ring_r[:12]:
            r_all = np.hypot(yy - cy, xx - cx)
            band = np.abs(r_all - rp) < 2.5
            iy, ix = np.nonzero(band)
            if iy.size < 400:
                continue
            v = img[iy, ix]
            base = np.percentile(v, 30)
            keep = v > base + 0.35 * (h)
            px, py = ix[keep].astype(float), iy[keep].astype(float)
            if px.size < 300:
                continue
            res = kasa(px, py)
            if res is None:
                continue
            kx, ky, kr = res
            for _ in range(2):
                rr = np.abs(np.hypot(px - kx, py - ky) - kr)
                sg = rr.std()
                m = rr < 2.5 * max(sg, 0.6)
                if m.sum() < 300:
                    break
                px, py = px[m], py[m]
                res = kasa(px, py)
                if res is None:
                    break
                kx, ky, kr = res
            if res is None:
                continue
            fits.append((kx, ky, kr, h, rr.std() if px.size else 9))
        if not fits:
            break
        # 只保留 σ 小的拟合，按峰高加权
        sig = np.array([f[4] for f in fits])
        good = sig < max(1.5, np.median(sig) + 1.0)
        fits = [f for f, g in zip(fits, good) if g]
        if not fits:
            break
        w = np.array([f[3] for f in fits], float)
        nx = np.sum([f[0] * wi for f, wi in zip(fits, w)]) / w.sum()
        ny = np.sum([f[1] * wi for f, wi in zip(fits, w)]) / w.sum()
        if dbg:
            print(f"  it{it}: ({cx:7.2f},{cy:7.2f}) -> ({nx:7.2f},{ny:7.2f}) "
                  f"rings={len(fits)} medσ={np.median(sig[good]):.2f}")
        cx = cx + damp * (nx - cx)
        cy = cy + damp * (ny - cy)
    return cx, cy


if __name__ == "__main__":
    s = 2
    img = load_pool(s)
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
        os.path.join(ZOO, "Mar3450.poni"))
    f2d = ai.getFit2D()
    gt = (f2d["centerX"] / s, f2d["centerY"] / s)
    rs = SC.gradient_vote_center(img)
    print("GT", tuple(round(v, 2) for v in gt), "RS", tuple(round(v, 2) for v in rs))
    for name, c0 in (("from GT", gt), ("from GT+20px", (gt[0] + 20, gt[1] - 15)),
                     ("from RS0", rs)):
        out = refine(img, c0)
        print(f"{name}: -> ({out[0]:.2f},{out[1]:.2f})  dGT="
              f"{np.hypot(out[0]-gt[0], out[1]-gt[1]):.2f}px")
