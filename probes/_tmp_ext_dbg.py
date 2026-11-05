# -*- coding: utf-8 -*-
"""_tmp_ext_dbg.py：极端箱 RS 失败解剖——GT/RS/几何中心三点位对比。"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
from torchfai.dataset import TorchFAIDataset
from torchfai.phases import PHASE_NAMES

PX = 344e-6


def bc(d, p1, p2, r1, r2):
    v2 = -d * np.tan(r1)
    v1 = np.tan(r2) * d / np.cos(r1)
    return (p2 + v2) / PX - 0.5, (p1 + v1) / PX - 0.5


def rs(img, sigma=3.0, pct=85.0):
    H, W = img.shape
    bg = gaussian_filter(img, 25.0)
    sm = gaussian_filter(img - bg, sigma)
    gx = ((sm[1:-1, 2:] - sm[1:-1, :-2]) * 0.5).ravel()
    gy = ((sm[2:, 1:-1] - sm[:-2, 1:-1]) * 0.5).ravel()
    yy, xx = np.mgrid[1:H - 1, 1:W - 1]
    xx = xx.ravel().astype(np.float64)
    yy = yy.ravel().astype(np.float64)
    g = np.hypot(gx, gy)
    pos = sm[1:-1, 1:-1].ravel()
    m = (g >= np.percentile(g, pct)) & (pos > 0)
    if m.sum() < 50:
        m = g >= np.percentile(g, 50)
    a, b, c = gy[m], -gx[m], gy[m] * xx[m] - gx[m] * yy[m]
    w = np.minimum(g[m], np.percentile(g[m], 99)) ** 2

    def solve(w):
        A11 = np.sum(w * a * a)
        A12 = np.sum(w * a * b)
        A22 = np.sum(w * b * b)
        B1 = np.sum(w * a * c)
        B2 = np.sum(w * b * c)
        det = A11 * A22 - A12 * A12
        if abs(det) < 1e-12 * max(A11 * A22, 1e-30):
            return None
        return ((A22 * B1 - A12 * B2) / det,
                (A11 * B2 - A12 * B1) / det)

    o = solve(w)
    if o is None:
        return None, m.sum(), float((g > 0).sum())
    r = a * o[0] + b * o[1] - c
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    keep = np.abs(r - np.median(r)) < 2.5 * s
    if keep.sum() > 50:
        o2 = solve(w * keep)
        if o2:
            o = o2
    return o, int(m.sum()), float(keep.mean())


ds = TorchFAIDataset(r"D:\pyfai-project\data\train_v4w3", "val")
H, W = ds._mimgs.shape[1], ds._mimgs.shape[2]
sel = ds._sel
p_all = ds._mparams[sel]
cx, cy = bc(p_all[:, 0], p_all[:, 1], p_all[:, 2], p_all[:, 3], p_all[:, 4])
off = np.maximum(np.abs(cx - (W - 1) / 2) / W,
                 np.abs(cy - (H - 1) / 2) / H)
rng = np.random.default_rng(11)
sample = rng.choice(len(sel), size=300, replace=False)
ext_idx = [i for i in sample if off[i] >= 0.45][:8]
print(f"{'GT(±)':>22}{'RS(±)':>22}{'RSerr':>7}{'几何err':>8}"
      f"{'掩膜px':>7}{'keep%':>6}{'相':>8}{'λ':>5}")
for i in ext_idx:
    img = np.asarray(ds._mimgs[int(sel[i])]).astype(np.float64)
    o, npx, kp = rs(img)
    aux = ds._maux[int(sel[i])]
    gc = np.hypot((W - 1) / 2 - cx[i], (H - 1) / 2 - cy[i])
    if o is None:
        print(f"({cx[i]:+7.0f},{cy[i]:+7.0f})  解退化  npx={npx}")
        continue
    print(f"({cx[i]:+7.0f},{cy[i]:+7.0f})"
          f"({o[0]:+7.0f},{o[1]:+7.0f})"
          f"{np.hypot(o[0]-cx[i], o[1]-cy[i]):7.0f}{gc:8.0f}"
          f"{npx:7d}{kp*100:6.0f}{PHASE_NAMES[int(aux[0])]:>8}{aux[5]:5.2f}")
