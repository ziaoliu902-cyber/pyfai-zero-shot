# -*- coding: utf-8 -*-
"""_tmp_rs_tune.py：RS 定心参数扫描——平滑σ/梯度门限/鲁棒两遍。"""
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter, convolve1d

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
from torchfai.dataset import TorchFAIDataset

PX = 344e-6


def bc(d, p1, p2, r1, r2):
    v2 = -d * np.tan(r1)
    v1 = np.tan(r2) * d / np.cos(r1)
    return (p2 + v2) / PX - 0.5, (p1 + v1) / PX - 0.5


def rs(img, sigma=2.0, pct=80.0, robust=True, bgsub=False, med=0):
    H, W = img.shape
    if med:
        from scipy.ndimage import median_filter
        img = median_filter(img, size=med)   # 杀孤立热点（环是延展结构，无损）
    if bgsub:
        bg = gaussian_filter(img, 25.0)
        img = img - bg                  # unsharp：消平背景/驼峰的常梯度偏置
    sm = gaussian_filter(img, sigma)
    gx = (sm[1:-1, 2:] - sm[1:-1, :-2]) * 0.5
    gy = (sm[2:, 1:-1] - sm[:-2, 1:-1]) * 0.5
    yy, xx = np.mgrid[1:H - 1, 1:W - 1]
    xx = xx.ravel().astype(np.float64)
    yy = yy.ravel().astype(np.float64)
    gx = gx.ravel()
    gy = gy.ravel()
    g = np.hypot(gx, gy)
    thr = np.percentile(g, pct)
    m = g >= thr
    if bgsub:
        m &= sm[1:-1, 1:-1].ravel() > 0     # 只保留环上（高于背景）像素
    if m.sum() < 50:                        # 空掩码设防（95 分位+bg 双门可清空）
        m = g >= np.percentile(g, 50)
    a = gy[m]
    b = -gx[m]
    c = gy[m] * xx[m] - gx[m] * yy[m]
    w = g[m]
    cap = np.percentile(w, 99.0)
    w = np.minimum(w, cap) ** 2

    def solve(w):
        A11 = np.sum(w * a * a)
        A12 = np.sum(w * a * b)
        A22 = np.sum(w * b * b)
        B1 = np.sum(w * a * c)
        B2 = np.sum(w * b * c)
        det = A11 * A22 - A12 * A12
        if not np.isfinite(det) or abs(det) < 1e-12 * max(A11 * A22, 1e-30):
            return (W - 1) / 2, (H - 1) / 2
        return ((A22 * B1 - A12 * B2) / det,
                (A11 * B2 - A12 * B1) / det)

    cx, cy = solve(w)
    if robust:
        r = a * cx + b * cy - c
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        keep = np.abs(r - np.median(r)) < 2.5 * s
        if keep.sum() > 50:
            cx, cy = solve(w * keep)
    return float(cx), float(cy)


ds = TorchFAIDataset(r"D:\pyfai-project\data\train_v4w3", "val")
H, W = ds._mimgs.shape[1], ds._mimgs.shape[2]
sel = ds._sel
p_all = ds._mparams[sel]
cx, cy = bc(p_all[:, 0], p_all[:, 1], p_all[:, 2], p_all[:, 3], p_all[:, 4])
off = np.maximum(np.abs(cx - (W - 1) / 2) / W,
                 np.abs(cy - (H - 1) / 2) / H)
rng = np.random.default_rng(11)
sample = rng.choice(len(sel), size=300, replace=False)
imgs = [np.asarray(ds._mimgs[int(sel[i])]).astype(np.float64)
        for i in sample]
gts = [(cx[i], cy[i]) for i in sample]
offs = off[sample]
mid = (offs >= 0.15) & (offs < 0.45)
ext = offs >= 0.45
near = offs < 0.15
for sigma, pct, bgs, md in ((3.0, 85, True, 0), (3.0, 85, True, 3),
                            (3.0, 85, True, 5), (1.5, 85, True, 5),
                            (5.0, 85, True, 5)):
    t0 = time.time()
    es = np.array([np.hypot(rs(im, sigma, pct, bgsub=bgs, med=md)[0] - g[0],
                            rs(im, sigma, pct, bgsub=bgs, med=md)[1] - g[1])
                   for im, g in zip(imgs, gts)])
    dt = (time.time() - t0) / len(imgs) * 1000
    print(f"σ={sigma:.1f} pct={pct:3.0f} bgsub={'Y' if bgs else 'N'} "
          f"med={md}: "
          f"近<{0.15}: med={np.median(es[near]):6.1f}  "
          f"中0.15~0.45: med={np.median(es[mid]):6.1f} "
          f"<10px={(es[mid]<10).mean()*100:3.0f}%  "
          f"极端: med={np.median(es[ext]):6.1f}  [{dt:.0f}ms]")
