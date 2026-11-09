# -*- coding: utf-8 -*-
"""_b12_fair2.py:Fairchild 原生 4096² 精拟合(2026-11-07)。

_b12_fair(画布 s=8,px_c=120µm)判决过线 Δ+2.61mm;本探针原生分辨率
(15µm)检测+拟合,判 Δ 能压到多少(目标 <2mm)。
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
print(f"原生 {img.shape} GT中心=({gt_cx:.1f},{gt_cy:.1f})")

bg = uniform_filter(img, 61)
resid = img - bg
v = resid.ravel()
mu = float(np.median(v))
sig = 1.4826 * float(np.median(np.abs(v - mu))) + 1e-12
mx = maximum_filter(img, size=5)
ys, xs = np.nonzero((img >= mx) & (resid > 6.0 * sig))
print(f"斑点:局部极大 {len(ys)}")
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
gap = np.min(np.abs(TTH[:, None] - TTH[None, :]) + np.eye(len(TTH)) * 9, axis=1)
keep = dt < 0.4 * gap[k]
d1, d2, jidx = sy[keep], sx[keep], k[keep]
cnt = np.bincount(jidx, minlength=len(TTH))
print(f"GT 指派:控制点 {keep.sum()}/{len(sx)} 环覆盖 {(cnt>0).sum()}/{len(TTH)}")

# 3σ 剔除离群点后拟合(斑点质心偶有粘连)
def fit5(d1, d2, jidx, p0):
    def resid(p):
        return fwd(p[0], p[1], p[2], p[3], p[4], d1, d2) - TTH[jidx]
    return least_squares(resid, p0, loss="soft_l1", f_scale=5e-4,
                         max_nfev=15000)

p0 = [D_MM * 1e-3, (gt_cy + 0.5) * PX, (gt_cx + 0.5) * PX, R1, R2]
f = fit5(d1, d2, jidx, p0)
# 精修:剔 |resid|>3×MAD 再拟
r = np.abs(f.fun)
med = np.median(r)
m2 = r < 3 * max(med, 2e-4)
f2 = fit5(d1[m2], d2[m2], jidx[m2], f.x)
Dm = f2.x[0] * 1e3
rr = np.hypot(d1[m2] - f2.x[1]/PX + 0.5, d2[m2] - f2.x[2]/PX + 0.5)
rms_px = float(np.sqrt(np.mean((f2.fun * rr / np.cos(TTH[jidx[m2]])**2)**2)))
print(f"\n原生拟合(剔 {len(d1)-m2.sum()} 离群,{m2.sum()}/{len(d1)} 点):")
print(f"  D={Dm:.2f}mm (GT {D_MM}, Δ{Dm-D_MM:+.2f}) "
      f"rot=({np.degrees(f2.x[3]):+.2f},{np.degrees(f2.x[4]):+.2f})°")
print(f"  中心=({f2.x[2]/PX-0.5:.0f},{f2.x[1]/PX-0.5:.0f}) "
      f"(GT ({gt_cx:.0f},{gt_cy:.0f})) rms={rms_px:.2f}px")
print(f"  判定:{'✓ Δ<2mm' if abs(Dm-D_MM) < 2.0 else ('△ Δ<5mm' if abs(Dm-D_MM)<5 else '✗')}")
