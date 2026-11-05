# -*- coding: utf-8 -*-
"""mar3450 亚像素定心实验。

思路：LaB6 @0.374Å 的内环很密（间距 5-7 画布 px），但外环（r>150px）间距
更大、更完整。先在 1/2 或 1/3 分辨率的画布上做"径向脊线 → Kasa 圆拟合"
迭代（每环独立拟合，按环质量加权），从梯度投票中心 (243,243) 收敛。
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
from scipy.ndimage import median_filter, map_coordinates, gaussian_filter

sys.path.insert(0, ROOT + r"\probes")
import solver_core as SC
from torchfai.phases import phase_qs


def load_native():
    img = np.asarray(fabio.open(os.path.join(ZOO, "LaB6_260210.mar3450")).data,
                     float)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    return img


def pool(img, s):
    H, W = img.shape
    h, w = H // s, W // s
    return img[:h * s, :w * s].reshape(h, s, w, s).mean(axis=(1, 3)), s


def radial(img, cx, cy, dr=0.5, rmax=None):
    H, W = img.shape
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - cy, xx - cx)
    if rmax is None:
        rmax = r.max()
    bins = np.arange(0, rmax + dr, dr)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    v = img.ravel()[ok]
    i = idx[ok]
    cnt = np.bincount(i, minlength=len(bins) - 1)
    prof = np.bincount(i, weights=v, minlength=len(bins) - 1) / np.maximum(cnt, 1)
    return bins[:-1] + dr / 2, prof


def ring_table(img, cx, cy, ai):
    r_ax, prof = radial(img, cx, cy, dr=0.5)
    bg = median_filter(prof, 31)
    d = prof - bg
    pk, props = find_peaks(d, prominence=0.02 * d.max(), distance=6)
    lam = ai.wavelength
    qs = np.asarray(phase_qs("LaB6"))
    tt = 2 * np.arcsin(np.clip(lam * qs / (4 * np.pi), 0, 0.9999))
    r_pred = ai.dist * np.tan(tt) / ai.detector.pixel1
    out = []
    for p in pk:
        r = r_ax[p]
        if r < 60:
            continue
        j = int(np.argmin(np.abs(r_pred - r)))
        out.append((r, d[p], r_pred[j], r_pred[j] - r))
    return out, r_ax, prof


if __name__ == "__main__":
    img = load_native()
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
        os.path.join(ZOO, "Mar3450.poni"))
    f2d = ai.getFit2D()
    gx, gy = f2d["centerX"], f2d["centerY"]
    print("native", img.shape, "GT center", gx, gy)
    rows, r_ax, prof = ring_table(img, gx, gy, ai)
    print(f"{'r_obs':>9}{'peak':>10}{'r_pred':>9}{'Δ':>8}")
    for r, h, rp, dlt in rows[:20]:
        print(f"{r:9.1f}{h:10.0f}{rp:9.1f}{dlt:+8.1f}")
