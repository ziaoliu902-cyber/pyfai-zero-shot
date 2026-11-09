# -*- coding: utf-8 -*-
"""_b12_fair5.py:Fairchild L2 终局——RANSAC 自洽子集(2026-11-07)。

_b12_fair4 诊断:控制点系统偏(GT 处 med 15px;[−180,−135)° 自交区
75px、线0/6 交叠环 34/78px;主力线2/4 仅 11-12px)。拟合 rms(9.75px)
<GT 处(54px)= 模型-数据不匹配下参数被拉走 +7mm。

策略:**RANSAC 自洽子集选择**——随机子集拟 5 参 → 数全体内点
(|dt|<tol)→ 内点集终拟。系统污染点因彼此不自洽被淘汰。
对照臂:人工干净子集(线∈{2,4},方位[90°,180°))。
判决:|ΔD|<2mm(目标 0.5)。
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
ph_all = np.degrees(np.arctan2(d1 - gt_cy, d2 - gt_cx))
print(f"指派内 {len(d1)} 点(RANSAC 池)")


def fit5(a, b, j, p0):
    def r_(p):
        return fwd(p[0], p[1], p[2], p[3], p[4], a, b) - TTH[j]
    return least_squares(r_, p0, max_nfev=12000)


p_gt = np.array([D_MM * 1e-3, (gt_cy + 0.5) * PX, (gt_cx + 0.5) * PX,
                 R1, R2])

# ---- 臂 0:干净子集(线2/4 + 方位[90,180)) ----
m_clean = np.isin(jj, (2, 4)) & (ph_all >= 90) & (ph_all < 180)
print(f"\n[臂0 干净子集] n={m_clean.sum()}")
f0 = fit5(d1[m_clean], d2[m_clean], jj[m_clean], p_gt)
D0 = f0.x[0] * 1e3
r0 = f0.fun * np.hypot(d1[m_clean]-f0.x[1]/PX+0.5, d2[m_clean]-f0.x[2]/PX+0.5) \
    / np.cos(TTH[jj[m_clean]]) ** 2
print(f"  D={D0:.3f}mm Δ{D0-D_MM:+.3f} rms={np.sqrt(np.mean(r0**2)):.2f}px"
      f"  {'✓' if abs(D0-D_MM)<2 else '✗'}")

# ---- 臂 1:RANSAC(随机 150 点拟 → 数内点 tol=2px → 内点集终拟) ----
rng = np.random.default_rng(3)
TOL_PX = 2.0
best = None
for it in range(24):
    sel = rng.choice(len(d1), 150, replace=False)
    try:
        fs = fit5(d1[sel], d2[sel], jj[sel], p_gt * np.array(
            [1.0, 1, 1, 1, 1]))
    except Exception:
        continue
    # 全体内点(换算 px)
    tt_m = fwd(fs.x[0], fs.x[1], fs.x[2], fs.x[3], fs.x[4], d1, d2)
    res = np.abs(tt_m - TTH[jj])
    rpx = res / np.cos(TTH[jj]) ** 2 * np.hypot(
        d1 - fs.x[1] / PX + 0.5, d2 - fs.x[2] / PX + 0.5)
    n_in = int((rpx < TOL_PX).sum())
    if best is None or n_in > best[0]:
        best = (n_in, fs.x.copy(), (rpx < TOL_PX))
print(f"\n[臂1 RANSAC] 最优假设内点 {best[0]}/{len(d1)} (tol {TOL_PX}px)")
m_in = best[2]
if m_in.sum() >= 150:
    f1 = fit5(d1[m_in], d2[m_in], jj[m_in], best[1])
    D1 = f1.x[0] * 1e3
    r1 = f1.fun * np.hypot(d1[m_in]-f1.x[1]/PX+0.5, d2[m_in]-f1.x[2]/PX+0.5) \
        / np.cos(TTH[jj[m_in]]) ** 2
    print(f"  内点终拟:n={m_in.sum()} D={D1:.3f}mm Δ{D1-D_MM:+.3f} "
          f"rms={np.sqrt(np.mean(r1**2)):.2f}px")
    print(f"  环覆盖 {np.unique(jj[m_in]).tolist()} "
          f"方位[{ph_all[m_in].min():.0f},{ph_all[m_in].max():.0f}]°")
    print(f"  判定:{'✓✓ Δ<0.5mm' if abs(D1-D_MM)<0.5 else ('✓ Δ<2mm' if abs(D1-D_MM)<2 else '✗')}")
