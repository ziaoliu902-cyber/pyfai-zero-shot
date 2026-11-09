# -*- coding: utf-8 -*-
"""_b12_fairchild.py:B12② Fairchild 真实数据判决(2026-11-07)。

生产 SOLVE FAIL 案例(束心出画布误判更正:GT 中心其实画布内 (2015,2996)
/4096;真难点=rot1=−29° 大倾角+部分弧)。

协议(与合成判决 v2 同型,检测层用真实斑点):
  1) 真实图 → canvas_of(default),斑点检测(5×5 局部极大+6σ 门+质心)
  2) 指派:GT 几何 fwd 算斑点 tth → 最近 Al2O3 线(<半线距)=GT 指派
     (指派层上界——序指派/RANSAC 是遗留;本判决分离拟合层)
  3) fit5(5 参共享 fwd 拟合):
     臂 a GT 指派+GT 初值(可解性上界)
     臂 b GT 指派+RS 粗中心/GT−10% D 初值(真实可得初值)
判决:|ΔD|<5mm(文档里程碑 2 目标)。

GT:D=88.207mm λ=1.54Å px=15µm rot1=−0.5068 rot2=+0.0167
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
from solver_core import canvas_of, center_radial_symmetry

PX = 15e-6
LAM = 1.540
D_MM = 88.2065
R1, R2 = -0.506767, 0.016707
P1_M, P2_M = 0.0302286, 0.0449458
QS = phase_qs("Al2O3")
TTH = 2 * np.arcsin(np.clip(QS * LAM / (4 * np.pi), 0, 0.9999))

img = fabio.open(ROOT + r"\data\real\Fairchild.edf").data.astype(np.float64)
canvas, s1f, _, _ = canvas_of(img, "default")
px_c = PX * s1f
H, W = canvas.shape
gt_cx = (P2_M / PX - 0.5) / s1f
gt_cy = (P1_M / PX - 0.5) / s1f
print(f"Fairchild:画布 {canvas.shape} s={s1f:.3f} px_c={px_c*1e6:.2f}µm "
      f"GT中心画布=({gt_cx:.1f},{gt_cy:.1f})  D={D_MM}mm rot1="
      f"{np.degrees(R1):.2f}° Al2O3 {len(QS)} 线,画布内线 "
      f"{int(((2*np.arctan(np.tan(TTH)*D_MM*1e3/PX*s1f))>0).sum())}+")

# ---- 真实斑点检测(_spot2d 核心) ----
im = canvas.astype(np.float64)
bg = uniform_filter(im, 61)
resid = im - bg
v = resid.ravel()
mu = float(np.median(v))
sig = 1.4826 * float(np.median(np.abs(v - mu))) + 1e-12
mx = maximum_filter(im, size=5)
ys, xs = np.nonzero((im >= mx) & (resid > 6.0 * sig))
print(f"斑点:局部极大 {len(ys)}")
sx, sy = [], []
for y, x in zip(ys, xs):
    y0, y1 = max(0, y - 2), min(H, y + 3)
    x0, x1 = max(0, x - 2), min(W, x + 3)
    w = np.clip(resid[y0:y1, x0:x1], 0, None)
    if w.sum() <= 0:
        continue
    gy_, gx_ = np.mgrid[y0:y1, x0:x1]
    sx.append((w * gx_).sum() / w.sum())
    sy.append((w * gy_).sum() / w.sum())
sx = np.array(sx); sy = np.array(sy)
print(f"斑点(质心):{len(sx)}")

# ---- GT 指派 ----
fwd = make_pyfai_fwd(px_c, px_c, None)
tth_sp = fwd(D_MM * 1e-3, (gt_cy + 0.5) * px_c, (gt_cx + 0.5) * px_c,
             R1, R2, sy + 0.0, sx + 0.0)
k = np.argmin(np.abs(tth_sp[:, None] - TTH[None, :]), axis=1)
dt = np.abs(tth_sp - TTH[k])
gap = np.min(np.abs(TTH[:, None] - TTH[None, :]) + np.eye(len(TTH)) * 9, axis=1)
keep = dt < 0.5 * gap[k]
d1 = sy[keep]           # 画布索引
d2 = sx[keep]
jidx = k[keep]
cnt = np.bincount(jidx, minlength=len(TTH))
print(f"GT 指派:控制点 {keep.sum()}/{len(sx)},环覆盖 "
      f"{(cnt>0).sum()}/{len(TTH)}(每环斑点 {cnt[cnt>0].min()}~"
      f"{cnt.max()})")

# ---- fit5 ----
def fit5(d1, d2, jidx, p0):
    def resid(p):
        return fwd(p[0], p[1], p[2], p[3], p[4], d1, d2) - TTH[jidx]
    return least_squares(resid, p0, loss="soft_l1", f_scale=5e-4,
                         max_nfev=12000)

rs = center_radial_symmetry(canvas)
print(f"RS 粗中心=({rs[0]:.0f},{rs[1]:.0f}) vs GT({gt_cx:.0f},{gt_cy:.0f}) "
      f"偏 {np.hypot(rs[0]-gt_cx, rs[1]-gt_cy):.0f}px")

for tag, p0 in (
    ("臂a GT初值", [D_MM*1e-3, (gt_cy+0.5)*px_c, (gt_cx+0.5)*px_c, R1, R2]),
    ("臂b RS中心+D−10%", [D_MM*0.9e-3, (rs[1]+0.5)*px_c, (rs[0]+0.5)*px_c,
                        R1*0.5, 0.0]),
):
    try:
        f = fit5(d1, d2, jidx, p0)
    except Exception as e:
        print(f"\n[{tag}] 拟合异常 {e}")
        continue
    Dm = f.x[0] * 1e3
    rr = np.hypot(d1 - f.x[1]/px_c + 0.5, d2 - f.x[2]/px_c + 0.5)
    rms_px = float(np.sqrt(np.mean((f.fun * rr / np.cos(TTH[jidx])**2)**2)))
    print(f"\n[{tag}]")
    print(f"  D={Dm:.2f}mm (GT {D_MM}, Δ{Dm-D_MM:+.2f}) "
          f"rot=({np.degrees(f.x[3]):+.2f},{np.degrees(f.x[4]):+.2f})° "
          f"(GT ({np.degrees(R1):+.2f},{np.degrees(R2):+.2f}))")
    print(f"  中心=({f.x[2]/px_c-0.5:.0f},{f.x[1]/px_c-0.5:.0f}) "
          f"(GT ({gt_cx:.0f},{gt_cy:.0f}))  rms={rms_px:.2f}px")
    print(f"  判定:{'✓ Δ<5mm' if abs(Dm-D_MM) < 5.0 else '✗ 未达'}")
