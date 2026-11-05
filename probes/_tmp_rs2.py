# -*- coding: utf-8 -*-
"""_tmp_rs2.py：两阶 RS——粗 RS → 提峰 → 环带掩膜 RS 精化。中箱目标 ≤5px。"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
from solver_core import ring_radii
from torchfai.dataset import TorchFAIDataset

PX = 344e-6


def bc(d, p1, p2, r1, r2):
    v2 = -d * np.tan(r1)
    v1 = np.tan(r2) * d / np.cos(r1)
    return (p2 + v2) / PX - 0.5, (p1 + v1) / PX - 0.5


def _rs_core(img, mask_sel, gx, gy, xx, yy, g):
    a = gy[mask_sel]
    b = -gx[mask_sel]
    c = gy[mask_sel] * xx[mask_sel] - gx[mask_sel] * yy[mask_sel]
    w = g[mask_sel]
    w = np.minimum(w, np.percentile(w, 99.0)) ** 2

    def solve(w):
        A11 = np.sum(w * a * a)
        A12 = np.sum(w * a * b)
        A22 = np.sum(w * b * b)
        B1 = np.sum(w * a * c)
        B2 = np.sum(w * b * c)
        det = A11 * A22 - A12 * A12
        if not np.isfinite(det) or abs(det) < 1e-12 * max(A11 * A22, 1e-30):
            return None
        return ((A22 * B1 - A12 * B2) / det,
                (A11 * B2 - A12 * B1) / det)

    out = solve(w)
    if out is None:
        return None
    cx, cy = out
    r = a * cx + b * cy - c
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    keep = np.abs(r - np.median(r)) < 2.5 * s
    if keep.sum() > 50:
        out2 = solve(w * keep)
        if out2 is not None:
            cx, cy = out2
    return cx, cy


def rs2(img, sigma=3.0, band=3.0):
    """两阶：全域粗 RS（bgsub+pct85）→ 环带掩膜精 RS。"""
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
    m = (g >= np.percentile(g, 85)) & (pos > 0)
    if m.sum() < 50:
        m = g >= np.percentile(g, 50)
    c1 = _rs_core(img, m, gx, gy, xx, yy, g)
    if c1 is None:
        return (W - 1) / 2, (H - 1) / 2, False
    cx0, cy0 = c1
    # 环带掩膜：峰半径 ±band 内且高于背景
    try:
        peaks = ring_radii(img, cx0, cy0, nb=384)[0]
    except Exception:
        return cx0, cy0, False
    if len(peaks) < 2:
        return cx0, cy0, False
    rp = np.array([r for r, _ in peaks])
    rpx = np.hypot(xx - cx0, yy - cy0)
    near_ring = np.zeros(len(rpx), bool)
    for r0 in rp:
        near_ring |= np.abs(rpx - r0) < band
    m2 = near_ring & (pos > 0) & (g > 0)
    if m2.sum() < 50:
        return cx0, cy0, False
    c2 = _rs_core(img, m2, gx, gy, xx, yy, g)
    if c2 is None:
        return cx0, cy0, False
    # 收敛保护：二阶若跳 >40px（伪峰环带）退回一阶
    if np.hypot(c2[0] - cx0, c2[1] - cy0) > 40:
        return cx0, cy0, False
    return c2[0], c2[1], True


ds = TorchFAIDataset(r"D:\pyfai-project\data\train_v4w3", "val")
H, W = ds._mimgs.shape[1], ds._mimgs.shape[2]
sel = ds._sel
p_all = ds._mparams[sel]
cx, cy = bc(p_all[:, 0], p_all[:, 1], p_all[:, 2], p_all[:, 3], p_all[:, 4])
off = np.maximum(np.abs(cx - (W - 1) / 2) / W,
                 np.abs(cy - (H - 1) / 2) / H)
rng = np.random.default_rng(11)
sample = rng.choice(len(sel), size=300, replace=False)
mid = (off[sample] >= 0.15) & (off[sample] < 0.45)
near = off[sample] < 0.15
ext = off[sample] >= 0.45
es, ok2 = [], []
import time
t0 = time.time()
for i in sample:
    img = np.asarray(ds._mimgs[int(sel[i])]).astype(np.float64)
    rcx, rcy, ok = rs2(img)
    es.append(np.hypot(rcx - cx[i], rcy - cy[i]))
    ok2.append(ok)
es = np.array(es)
ok2 = np.array(ok2)
print(f"两阶 RS {len(sample)} 样本 {(time.time()-t0)/len(sample)*1000:.0f}ms/帧，"
      f"二阶触发率 {ok2.mean()*100:.0f}%")
for lab, m in (("近<0.15", near), ("中0.15~0.45", mid), ("极端", ext)):
    if m.sum() < 5:
        continue
    e = es[m]
    print(f"  {lab}: n={m.sum():3d} 中位={np.median(e):6.1f}px  "
          f"<10px={(e<10).mean()*100:3.0f}%  <5px={(e<5).mean()*100:3.0f}%")
