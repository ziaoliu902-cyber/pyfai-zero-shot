# -*- coding: utf-8 -*-
"""_b12_fair3.py:B12② L2 指派层——Fairchild 原生精化(2026-11-07)。

_b12_fair2:原生 4096² GT 指派 951 点 → Δ+7.19mm,rms 9.38px——
**指派污染**(1947 斑点中弱伪点/粘连质心把解拉走;画布 s=8 版反而
Δ+2.61)。

三件套(本探针网格):
  A 强度门:斑点残差峰高 ≥ kσ ×{6,8,10} + 积分强度分位 {—,p50};
  B 指派收紧:dt < {0.4, 0.25} × gap;
  C 迭代剔除:fit5 → 剔 |resid|>3×MAD → refit(≤5 轮,点数下限 150)。
判决:|ΔD| < 2mm(目标),理想 <0.5mm;报告每配置轨迹。

GT:D=88.2065mm λ=1.54Å px=15µm rot1=−0.5068 rot2=+0.0167 Al2O3。
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

# ---- 斑点检测(一次,记录强度) ----
bg = uniform_filter(img, 61)
resid = img - bg
v = resid.ravel()
mu = float(np.median(v))
sig = 1.4826 * float(np.median(np.abs(v - mu))) + 1e-12
mx = maximum_filter(img, size=5)
ys, xs = np.nonzero((img >= mx) & (resid > 6.0 * sig))
spot_i, spot_x, spot_y, spot_h = [], [], [], []
for y, x in zip(ys, xs):
    y0, y1 = max(0, y - 3), min(H, y + 4)
    x0, x1 = max(0, x - 3), min(W, x + 4)
    w = np.clip(resid[y0:y1, x0:x1], 0, None)
    if w.sum() <= 0:
        continue
    gy_, gx_ = np.mgrid[y0:y1, x0:x1]
    spot_x.append((w * gx_).sum() / w.sum())
    spot_y.append((w * gy_).sum() / w.sum())
    spot_h.append(resid[y, x])
    spot_i.append(w.sum())
spot_x = np.array(spot_x); spot_y = np.array(spot_y)
spot_h = np.array(spot_h); spot_i = np.array(spot_i)
print(f"检测:{len(spot_x)} 斑点(6σ 底门),强度 "
      f"p50={np.percentile(spot_i,50):.0f} p75={np.percentile(spot_i,75):.0f}")

fwd = make_pyfai_fwd(PX, PX, None)


def fit5(d1, d2, jidx, p0):
    def resid_(p):
        return fwd(p[0], p[1], p[2], p[3], p[4], d1, d2) - TTH[jidx]
    return least_squares(resid_, p0, loss="soft_l1", f_scale=5e-4,
                         max_nfev=15000)


def run(kk, dt_rel, ipct):
    m = spot_h >= kk * sig
    if ipct is not None:
        m &= spot_i >= np.percentile(spot_i[m], ipct * 100)
    sx, sy = spot_x[m], spot_y[m]
    tth_sp = fwd(D_MM * 1e-3, (gt_cy + 0.5) * PX, (gt_cx + 0.5) * PX,
                 R1, R2, sy + 0.0, sx + 0.0)
    k = np.argmin(np.abs(tth_sp[:, None] - TTH[None, :]), axis=1)
    dt = np.abs(tth_sp - TTH[k])
    gap = np.min(np.abs(TTH[:, None] - TTH[None, :])
                 + np.eye(len(TTH)) * 9, axis=1)
    keep = dt < dt_rel * gap[k]
    d1, d2, jj = sy[keep], sx[keep], k[keep]
    if len(d1) < 150:
        return None
    p = np.array([D_MM * 1e-3, (gt_cy + 0.5) * PX, (gt_cx + 0.5) * PX,
                  R1, R2])
    traj = []
    for it in range(5):
        f = fit5(d1, d2, jj, p)
        r = np.abs(f.fun)
        mad = 1.4826 * np.median(r)
        m2 = r <= 3.0 * max(mad, 1e-4)
        traj.append((len(d1), f.x[0] * 1e3,
                     float(np.sqrt(np.mean(f.fun ** 2)))))
        if m2.all() or m2.sum() < 150:
            break
        d1, d2, jj, p = d1[m2], d2[m2], jj[m2], f.x
    Dm = f.x[0] * 1e3
    rr = np.hypot(d1 - f.x[1] / PX + 0.5, d2 - f.x[2] / PX + 0.5)
    rms_px = float(np.sqrt(np.mean((f.fun * rr / np.cos(TTH[jj]) ** 2) ** 2)))
    return Dm, rms_px, traj, f


print(f"\n{'配置(kσ/dt/int)':22} {'n→':>10} {'D':>8} {'Δ':>8} {'rms_px':>7}")
best = None
for kk in (6.0, 8.0, 10.0):
    for dt_rel, ipct in ((0.4, None), (0.25, None), (0.4, 50)):
        out = run(kk, dt_rel, ipct)
        tag = f"{kk:.0f}σ/dt{dt_rel}/{'p50' if ipct else '—'}"
        if out is None:
            print(f"{tag:22} 点数不足")
            continue
        Dm, rms_px, traj, f = out
        d = abs(Dm - D_MM)
        print(f"{tag:22} {traj[0][0]:>5}→{traj[-1][0]:<4} {Dm:8.2f} "
              f"{Dm-D_MM:+8.2f} {rms_px:7.2f}"
              + ("  ←" if (best is None or d < best[0]) else ""))
        if best is None or d < best[0]:
            best = (d, tag, out)

d, tag, (Dm, rms_px, traj, f) = best
print(f"\n最优 {tag}:D={Dm:.3f}mm Δ{Dm-D_MM:+.3f}mm rms={rms_px:.2f}px")
print(f"  轨迹(点数,D,rms): {[(n, round(dd,2), round(rr,5)) for n,dd,rr in traj]}")
print(f"  判定:{'✓ Δ<2mm' if d < 2 else '✗ 未达'}"
      f"{' ✓✓ Δ<0.5mm' if d < 0.5 else ''}")
