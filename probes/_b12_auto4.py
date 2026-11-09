# -*- coding: utf-8 -*-
"""_b12_auto4.py:B12② 半自动契约边界测定(2026-11-08)。

背景:v4 fold+ICP 在 Fairchild 上臂1(中心偏(200,−150),D−10%)
Δ+0.21mm ✓、臂2(反向偏)✗——初值方向敏感(y=rot1 敏感轴)。
v5/v6 盲指派三连败已封档;半自动(用户初值)是实证形态。

本探针:测定 v4 流水(fold 找 rot + ICP)的**初值契约**:
  中心偏 ±100px(两向)× D ±5%(现实用户:目测束心/记错 D)
判决:全过 → 契约=中心±100px/D±5%,落地 solve_partial API;
     部分 → 记录边界,API 标注置信度返回。
"""
import os
import sys

import numpy as np
import fabio
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.optimize import least_squares

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from poni_out import make_pyfai_fwd
from torchfai.phases import phase_qs

PX_ = 15e-6
GRID_R = np.deg2rad([0, 10, 20, 30, 40])
TOLS = (0.25, 0.15, 0.08)


def icp(fwd, d1, d2, tth_q, p, tols, min_n=30, d_range=(10e-3, 2.0)):
    lo = [d_range[0], p[1] - 1000 * PX_, p[2] - 1000 * PX_,
          np.deg2rad(-60), np.deg2rad(-60)]
    hi = [d_range[1], p[1] + 1000 * PX_, p[2] + 1000 * PX_,
          np.deg2rad(60), np.deg2rad(60)]
    p = np.clip(np.array(p, float), lo, hi)
    m = None
    for tol in tols:
        tm = fwd(p[0], p[1], p[2], p[3], p[4], d1, d2)
        k = np.argmin(np.abs(tm[:, None] - tth_q[None, :]), axis=1)
        dtt = np.abs(tm - tth_q[k])
        gap = np.min(np.abs(tth_q[:, None] - tth_q[None, :])
                     + np.eye(len(tth_q)) * 9, axis=1)
        m = dtt < tol * gap[k]
        if m.sum() < min_n:
            return None, 0, np.inf
        a, b, j = d1[m], d2[m], k[m]

        def r_(pp):
            return fwd(pp[0], pp[1], pp[2], pp[3], pp[4], a, b) - tth_q[j]
        try:
            f = least_squares(r_, p, bounds=(lo, hi), max_nfev=8000)
        except Exception:
            return None, 0, np.inf
        p = f.x
    rr = np.hypot(a - p[1] / PX_ + 0.5, b - p[2] / PX_ + 0.5)
    rms = float(np.sqrt(np.mean((f.fun * rr / np.cos(tth_q[j]) ** 2) ** 2)))
    return (p, int(m.sum()), rms), int(m.sum()), rms


def spots_of(img, ksig=6.0):
    H, W = img.shape
    bg = uniform_filter(img, 61)
    resid = img - bg
    v = resid.ravel()
    mu = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - mu))) + 1e-12
    mx = maximum_filter(img, size=5)
    ys, xs = np.nonzero((img >= mx) & (resid > ksig * sig))
    sx, sy = [], []
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 3), min(H, y + 4)
        x0, x1 = max(0, x - 3), min(W, x + 4)
        w = np.clip(resid[y0:y1, x0:x1], 0, None)
        if w.sum() <= 0:
            continue
        gy_, gx_ = np.mgrid[y0:y1, x0:x1]
        sx.append((w * gx_).sum() / w.sum())
        sy.append((w * gy_).sum() / w.sum())
    return np.array(sx), np.array(sy)


def fold_icp(sx, sy, px, lam, qs, cx0, cy0, D0):
    """v4 两阶段 fold(rot 网格+中心细扫)+ ICP。D0 mm。"""
    global PX_
    PX_ = px
    fwd = make_pyfai_fwd(px, px, None)
    tth_q = 2 * np.arcsin(np.clip(qs * lam / (4 * np.pi), 0, 0.9999))
    d1, d2 = sy, sx
    rng_d = (D0 * 1e-3 * 0.3, D0 * 1e-3 * 3.0)
    edges = np.arange(0, tth_q[-1] * 1.15, 2e-4)
    ker = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)

    def score(p):
        tm = fwd(p[0], p[1], p[2], p[3], p[4], d1, d2)
        tm = tm[(tm > 0.02) & (tm < edges[-1])]
        if len(tm) < 30:
            return 0.0
        h, _ = np.histogram(tm, bins=edges)
        h = np.convolve(h, ker, mode="same")
        idx = np.clip(np.searchsorted(edges, tth_q) - 1, 0, len(h) - 1)
        return float(h[idx].sum())

    best = None
    for dx in (-150, 0, 150):
        for dy in (-150, 0, 150):
            for r1 in list(GRID_R) + list(-GRID_R[1:]):
                for fd in (0.9, 1.0, 1.1):
                    p0 = [D0 * fd * 1e-3, (cy0 + dy + 0.5) * px,
                          (cx0 + dx + 0.5) * px, r1, 0.0]
                    s = score(p0)
                    if best is None or s > best[0]:
                        best = (s, p0)
    s, p0 = best
    cxg, cyg = p0[2] / px - 0.5, p0[1] / px - 0.5
    for ddx in range(-6, 7):
        for ddy in range(-6, 7):
            q = [p0[0], (cyg + ddy * 25 + 0.5) * px,
                 (cxg + ddx * 25 + 0.5) * px, p0[3], 0.0]
            s2 = score(q)
            if s2 > best[0]:
                best = (s2, q)
    out = icp(fwd, d1, d2, tth_q, list(best[1]), TOLS, d_range=rng_d)
    if out[0] is None:
        return None
    return out[0]


PX = 15e-6
PX_ = PX
LAM = 1.540
D_GT = 88.2065
gt_cx = (0.0449458 / PX) - 0.5
gt_cy = (0.0302286 / PX) - 0.5
QS = phase_qs("Al2O3")
img = fabio.open(ROOT + r"\data\real\Fairchild.edf").data.astype(np.float64)
sx, sy = spots_of(img)
print("半自动契约测定:Fairchild,中心 ±100px × D ±5%,fold 找 rot")
print("=" * 62)
n_ok = 0
for tag, dx, dy, fD in (("中心(+100,−100) D−5%", 100, -100, 0.95),
                        ("中心(+100,−100) D+5%", 100, -100, 1.05),
                        ("中心(−100,+100) D−5%", -100, 100, 0.95),
                        ("中心(−100,+100) D+5%", -100, 100, 1.05)):
    res = fold_icp(sx, sy, PX, LAM, QS, gt_cx + dx, gt_cy + dy,
                   D_GT * fD)
    if res is None:
        print(f"\n[{tag}] ICP 失败(内点<30)")
        continue
    p, n, rms = res
    Df = p[0] * 1e3
    ok = abs(Df - D_GT) < 5.0
    n_ok += ok
    print(f"\n[{tag}] 内点 {n} rms {rms:.1f}px")
    print(f"  D={Df:.2f}mm Δ{Df-D_GT:+.2f} "
          f"rot=({np.degrees(p[3]):+.1f},{np.degrees(p[4]):+.1f})° "
          f"中心=({p[2]/PX-0.5:.0f},{p[1]/PX-0.5:.0f}) "
          f"{'✓' if ok else '✗'}")
print(f"\n判决:{n_ok}/4 过(契约=中心±100px+D±5%)"
      + ("→ 可落地 solve_partial" if n_ok == 4 else "→ 边界更紧/带置信度"))
