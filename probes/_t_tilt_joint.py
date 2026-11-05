# -*- coding: utf-8 -*-
"""_t_tilt_joint.py：倾角联合 λ–D 拟合探针（破简并之战，2026-11-01）。

物理：小角域 r∝D·λ·q 只有简并积；宽角 tan(2asin) 非线性弱破简并，但
1D 谱被倾角 r(χ) 调制毒化（T1 实证 σ=44% 拒收）。联合 (λ, dist, poni,
rot1, rot2) 把系统调制吸进倾角参数，剩下的环间非线性才给 λ 证据。

判据：
  ① σ(λ)/λ（JᵀJ 反演）是否 <2%（T1 简并门）；
  ② 固定 λ 扫描的 rms(λ) 曲线是否有清晰极小（λ 可辨）还是平坦（简并）；
  ③ λ̂D̂ 积是否守恒 ≈83.2-83.4（物理一致）。
GT（Mar3450.poni 名义）：λ=0.3738Å D=222.55mm 积=83.20。
"""
import os
import sys
import time
import numpy as np
from scipy.optimize import least_squares

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import fabio
import pyFAI.azimuthalIntegrator
from solver_core import solve_auto, to_canvas
from poni_out import sector_peaks, make_pyfai_fwd
from torchfai.phases import phase_qs

img = np.asarray(fabio.open(os.path.join(ZOO, "LaB6_260210.mar3450")).data,
                 float)
img[img >= 1e9] = 0.0
img[img < 0] = 0.0
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload("Mar3450.poni")
f2d = ai.getFit2D()
gt_cx, gt_cy = f2d["centerX"], f2d["centerY"]
gt_D, gt_lam = ai.dist * 1000, ai.wavelength * 1e10
print(f"GT: λ={gt_lam:.4f}Å D={gt_D:.2f}mm rot=({ai.rot1:.5f},{ai.rot2:.5f})"
      f" 积={gt_lam*gt_D:.2f}")

canvas, s, _, _ = to_canvas(img, max_side=1725)
px_m = float(ai.detector.pixel1) * s          # 画布物理 px = 200µm
px_nat = float(ai.detector.pixel1)
lam0 = 0.3611                                  # fallback 网格 λ̂（简并脊点）

# ---- ① 2D 精化解作初始（fallback 胜出格点）----
t0 = time.perf_counter()
r = solve_auto(canvas, lam0, px_m, cascade=True, auto_center=True,
               refine_2d=True)
ph, sol = r
cx, cy, D_px = sol[0], sol[1], sol[2]
print(f"\n初始（s={s} λ0={lam0}, {time.perf_counter()-t0:.0f}s）: {ph} "
      f"c=({cx:.2f},{cy:.2f}) D={D_px*px_m*1000:.2f}mm "
      f"dc={np.hypot(cx*s-gt_cx, cy*s-gt_cy):.2f}px "
      f"r2d={sol[10] if len(sol)>=11 else 0}环")

# ---- ② 控制点：12 环 × 36 扇区 ----
qs = np.asarray(phase_qs(ph), float)
tt0 = 2 * np.arcsin(np.clip(qs * lam0 / (4 * np.pi), 0, 0.999))
radii0 = D_px * np.tan(tt0)
pts = sector_peaks(canvas, cx, cy, radii0, nsec=36, win=6.0,
                   mass_rel=0.10, bump=1.5)
ri = np.array([p[0] for p in pts])
rp = np.array([p[2] for p in pts])
ms = np.array([p[3] for p in pts])
rows = cy + rp * np.sin(np.array([p[1] for p in pts]))
cols = cx + rp * np.cos(np.array([p[1] for p in pts]))
d1f = (rows + 0.5) * s - 0.5                   # 原生像素索引
d2f = (cols + 0.5) * s - 0.5
tt_t = tt0[ri]
wgt = np.sqrt(ms); wgt = wgt / wgt.mean()
print(f"控制点: {len(pts)}pt / {len(set(ri))}环  "
      f"r∈[{rp.min():.0f},{rp.max():.0f}]px")

fwd = make_pyfai_fwd(px_nat, px_nat, None)     # 原生物理 px
dist0 = D_px * px_m
poni1_0 = (cy + 0.5) * px_m
poni2_0 = (cx + 0.5) * px_m


def resid6(p):
    tt_mod = fwd(p[1], p[2], p[3], p[4], p[5], d1f, d2f)
    tt_tar = 2 * np.arcsin(np.clip(qs[ri] * p[0] / (4 * np.pi), 0, 0.999))
    return (tt_mod - tt_tar) * wgt


# ---- ③ 联合 6 参（λ 自由）----
p0 = [lam0, dist0, poni1_0, poni2_0, 0.0, 0.0]
t0 = time.perf_counter()
res6 = least_squares(resid6, p0, loss="soft_l1", f_scale=1.5e-3,
                     x_scale=[0.01, 0.005, 0.005, 0.005, 0.01, 0.01],
                     bounds=([lam0 * 0.90, dist0 - 0.03,
                              poni1_0 - 40 * px_m, poni2_0 - 40 * px_m,
                              -0.06, -0.06],
                             [lam0 * 1.10, dist0 + 0.03,
                              poni1_0 + 40 * px_m, poni2_0 + 40 * px_m,
                              0.06, 0.06]))
lam_h, dist_h = float(res6.x[0]), float(res6.x[1] * 1000)
J = np.asarray(res6.jac); n = len(tt_t)
s2 = max(2.0 * res6.cost / max(n - 6, 1), 1e-18)
try:
    cov = np.linalg.inv(J.T @ J) * s2
    sig_lam = float(np.sqrt(max(cov[0, 0], 0)) / lam_h)
except Exception:
    sig_lam = None
rms6 = float(np.sqrt(np.mean((resid6(res6.x) / wgt) ** 2)))
print(f"\n[联合6参] λ̂={lam_h:.4f}σ={sig_lam*100 if sig_lam else -1:.1f}% "
      f"D̂={dist_h:.2f}mm(Δ{dist_h-gt_D:+.2f}) 积={lam_h*dist_h:.2f} "
      f"rot=({res6.x[4]:.5f},{res6.x[5]:.5f}) "
      f"rms={rms6*1000:.2f}mrad ({time.perf_counter()-t0:.1f}s)")

# ---- ④ 固定 λ 曲线：rms(λ) 有无极小 ----
print("\n[固定λ 5参] λ → D̂ / 积 / rms(mrad)：")
for lam_fix in (0.3551, 0.3611, 0.3660, 0.3705, 0.3738, 0.3796, 0.3860):
    def resid5(p):
        tt_mod = fwd(p[0], p[1], p[2], p[3], p[4], d1f, d2f)
        tt_tar = 2 * np.arcsin(np.clip(qs[ri] * lam_fix / (4 * np.pi), 0,
                                       0.999))
        return (tt_mod - tt_tar) * wgt
    r5 = least_squares(resid5, [dist0, poni1_0, poni2_0, 0.0, 0.0],
                       loss="soft_l1", f_scale=1.5e-3,
                       x_scale=[0.005, 0.005, 0.005, 0.01, 0.01],
                       bounds=([dist0 - 0.03, poni1_0 - 40 * px_m,
                                poni2_0 - 40 * px_m, -0.06, -0.06],
                               [dist0 + 0.03, poni1_0 + 40 * px_m,
                                poni2_0 + 40 * px_m, 0.06, 0.06]))
    rms5v = float(np.sqrt(np.mean((resid5(r5.x) / wgt) ** 2)))
    print(f"  λ={lam_fix:.4f}: D̂={r5.x[0]*1000:7.2f}mm "
          f"积={lam_fix*r5.x[0]*1000:.2f} rms={rms5v*1000:6.2f} "
          f"rot=({r5.x[3]:+.4f},{r5.x[4]:+.4f})")
