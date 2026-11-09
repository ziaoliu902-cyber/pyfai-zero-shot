# -*- coding: utf-8 -*-
"""_b12_fair4.py:Fairchild 控制点质量诊断(2026-11-07)。

_b12_fair3:迭代剔除后 871 点仍 Δ+7.15mm rms 9.75px——污染是系统性的。
诊断:在 **GT 参数处**(不拟合)算残差分布:
  - 若 GT 处 rms 也 ~10px → 控制点本身偏(质心/粘连/伪点),修检测;
  - 若 GT 处 rms 小(~1px)→ 拟合被拖走(权重/多极小),修拟合。
分环/分方位直方 + 残差 vs 强度散点(找污染来源)。
"""
import os
import sys

import numpy as np
import fabio
from scipy.ndimage import maximum_filter, uniform_filter

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
sx, sy, sh = [], [], []
for y, x in zip(ys, xs):
    y0, y1 = max(0, y - 3), min(H, y + 4)
    x0, x1 = max(0, x - 3), min(W, x + 4)
    w = np.clip(resid[y0:y1, x0:x1], 0, None)
    if w.sum() <= 0:
        continue
    gy_, gx_ = np.mgrid[y0:y1, x0:x1]
    sx.append((w * gx_).sum() / w.sum())
    sy.append((w * gy_).sum() / w.sum())
    sh.append(resid[y, x])
sx = np.array(sx); sy = np.array(sy); sh = np.array(sh)

fwd = make_pyfai_fwd(PX, PX, None)
tth_sp = fwd(D_MM * 1e-3, (gt_cy + 0.5) * PX, (gt_cx + 0.5) * PX,
             R1, R2, sy + 0.0, sx + 0.0)
k = np.argmin(np.abs(tth_sp[:, None] - TTH[None, :]), axis=1)
dt = np.abs(tth_sp - TTH[k])
gap = np.min(np.abs(TTH[:, None] - TTH[None, :]) + np.eye(len(TTH)) * 9,
             axis=1)
keep = dt < 0.4 * gap[k]
print(f"指派内 {keep.sum()}/{len(sx)} 点;GT 参数处残差(dt)统计:")
dt_k = dt[keep]
print(f"  全体: rms(px)={np.sqrt(np.mean((dt_k*1e3/15*1e0)**2))*0+np.sqrt(np.mean((dt_k/np.cos(TTH[k[keep]])**2 * np.hypot(sx[keep]-gt_cx, sy[keep]-gt_cy))**2)):.2f}"
      f"  中位={np.median(dt_k)*1e3:.3f}mrad")

# 换算 px:dt(rad) × dr/dθ ≈ dt × r/cos²(2θ)(r 用到 GT 中心距离)
r_px = np.hypot(sx[keep] - gt_cx, sy[keep] - gt_cy)
res_px = dt_k / np.cos(TTH[k[keep]]) ** 2 * r_px
print(f"  换算 px:rms={np.sqrt(np.mean(res_px**2)):.2f} "
      f"中位={np.median(res_px):.2f} p90={np.percentile(res_px,90):.2f}")
print("\n分环(线 j:点数 / 残差中位 px / p90 px):")
for j in range(len(TTH)):
    m = k[keep] == j
    if m.sum() < 5:
        continue
    print(f"  线{j:2} 2θ={np.degrees(TTH[j]):5.2f}° n={m.sum():4} "
          f"med={np.median(res_px[m]):7.2f} p90={np.percentile(res_px[m],90):8.2f}")
ph = np.degrees(np.arctan2(sy[keep] - gt_cy, sx[keep] - gt_cx))
print("\n分方位(45° 一格:点数 / 残差中位 px):")
for a0 in range(-180, 180, 45):
    m = (ph >= a0) & (ph < a0 + 45)
    if m.sum() < 5:
        continue
    print(f"  [{a0:4},{a0+45:4})° n={m.sum():4} med={np.median(res_px[m]):7.2f}")
print("\n残差 vs 斑点强度(强/弱分半):")
med_h = np.median(sh[keep])
for lbl, m in (("强(p50+)", sh[keep] >= med_h), ("弱", sh[keep] < med_h)):
    print(f"  {lbl}: n={m.sum()} med={np.median(res_px[m]):.2f}px "
          f"p90={np.percentile(res_px[m],90):.2f}px")
