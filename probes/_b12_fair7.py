# -*- coding: utf-8 -*-
"""_b12_fair7.py:Fairchild L2 终局2——块平差 nuisance 参数(2026-11-07)。

fair6 教训:块剔除不稳定(轮1 漂 +4.35)——污染是连续系统场,不是
离散坏块;"好"块也有 ~8px 系统残差且块间方向不一致,拟合在块间折衷,
解漂移。

**块平差**:残差模型 tth_model(p_geo) + δ_b(块) − tth_q,
δ_b = 每块(环×30°方位) nuisance 偏移(|δ_b|≤8px 换算),吸收系统场
(环自交/粘连/畸变);几何 5 参由**块间相对**信息(不同环不同方位的
δ 交叉约束)决定。δ_b 不进结果。
判决:|ΔD|<2mm;对照 fair5 臂0(Δ+1.68)。
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

PX = 15e-6
LAM = 1.540
D_MM = 88.2065
R1, R2 = -0.506767, 0.016707
P1_M, P2_M = 0.0302286, 0.0449458
QS = phase_qs("Al2O3")
TTH = 2 * np.arcsin(np.clip(QS * LAM / (4 * np.pi), 0, 0.9999))

img = fabio.open(ROOT + r"\data\real\Fairchild.edf").data.astype(np.float64)
H, W = img.shape
gt_cx = P2_M / PX - 0.5
gt_cy = P1_M / PX - 0.5

bg = uniform_filter(img, 61)
resid = img - bg
v = resid.ravel()
mu = float(np.median(v))
sig = 1.4826 * float(np.median(np.abs(v - mu))) + 1e-12
mx = maximum_filter(img, size=5)
ys, xs = np.nonzero((img >= mx) & (resid > 6.0 * sig))
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
sx = np.array(sx); sy = np.array(sy)

fwd = make_pyfai_fwd(PX, PX, None)
tth_sp = fwd(D_MM * 1e-3, (gt_cy + 0.5) * PX, (gt_cx + 0.5) * PX,
             R1, R2, sy + 0.0, sx + 0.0)
k = np.argmin(np.abs(tth_sp[:, None] - TTH[None, :]), axis=1)
dt = np.abs(tth_sp - TTH[k])
gap = np.min(np.abs(TTH[:, None] - TTH[None, :]) + np.eye(len(TTH)) * 9,
             axis=1)
keep = dt < 0.4 * gap[k]
d1, d2, jj = sy[keep], sx[keep], k[keep]
ph = np.degrees(np.arctan2(d1, d2))
blk = (jj * 100 + np.clip((ph // 30).astype(int), -6, 5))
ub, ib = np.unique(blk, return_inverse=True)
B = len(ub)
cnts = np.bincount(ib)
big = cnts[ib] >= 8
d1, d2, jj, ib = d1[big], d2[big], jj[big], ib[big]
ub, ib = np.unique(ib, return_inverse=True)
B = len(ub)
print(f"池 {len(d1)} 点 / {B} 块(环×30°,≥8 点)")


def rpx(r, a, b, j):
    return r * np.hypot(a, b) / np.cos(TTH[j]) ** 2


def fit_block(d1, d2, jj, ib, B, p0, use_delta=True):
    tt = TTH

    def r_(p):
        pg = p[:5]
        m = fwd(pg[0], pg[1], pg[2], pg[3], pg[4], d1, d2) - tt[jj]
        if use_delta:
            m = m + p[5:][ib]
        return m
    p0 = np.append(p0, np.zeros(B))
    lo = np.array([-np.inf] * 5 + [-4e-3] * B)
    hi = np.array([np.inf] * 5 + [4e-3] * B)
    return least_squares(r_, p0, bounds=(lo, hi), max_nfev=15000)


p0 = np.array([D_MM * 1e-3, (gt_cy + 0.5) * PX, (gt_cx + 0.5) * PX, R1, R2])

# 对照:无 δ(=fair2 拟合)
f_no = fit_block(d1, d2, jj, ib, B, p0, use_delta=False)
r_no = rpx(f_no.fun, d1 - f_no.x[1] / PX + 0.5,
           d2 - f_no.x[2] / PX + 0.5, jj)
print(f"\n[无 δ 对照] D={f_no.x[0]*1e3:.3f}mm Δ{f_no.x[0]*1e3-D_MM:+.3f} "
      f"rms={np.sqrt(np.mean(r_no**2)):.2f}px")

# 块平差
f = fit_block(d1, d2, jj, ib, B, p0)
Dm = f.x[0] * 1e3
dpx = np.hypot(d1 - f.x[1] / PX + 0.5, d2 - f.x[2] / PX + 0.5)
r_full = rpx(f.fun, d1 - f.x[1] / PX + 0.5, d2 - f.x[2] / PX + 0.5, jj)
delta_px = f.x[5:][ib] * dpx / np.cos(TTH[jj]) ** 2
print(f"\n[块平差] D={Dm:.3f}mm Δ{Dm-D_MM:+.3f}mm "
      f"rms(含δ)={np.sqrt(np.mean(r_full**2)):.2f}px")
r_geo = rpx(f.fun - f.x[5:][ib], d1 - f.x[1] / PX + 0.5,
            d2 - f.x[2] / PX + 0.5, jj)
print(f"  δ 幅度:med={np.median(np.abs(delta_px)):.1f}px "
      f"p90={np.percentile(np.abs(delta_px),90):.1f}px "
      f"(上限 8px 等效)")
print(f"  rot=({np.degrees(f.x[3]):+.2f},{np.degrees(f.x[4]):+.2f})° "
      f"中心=({f.x[2]/PX-0.5:.0f},{f.x[1]/PX-0.5:.0f})")
print(f"  判定:{'✓✓ Δ<0.5mm' if abs(Dm-D_MM)<0.5 else ('✓ Δ<2mm' if abs(Dm-D_MM)<2 else '✗')}")
