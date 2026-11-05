# -*- coding: utf-8 -*-
"""_t1_lsq_dbg.py：P1M 上直接调 refine_lambda_d（dbg）看匹配数、λ 走向、σ。"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import fabio
import pyFAI.azimuthalIntegrator
from torchfai.phases import phase_qs
import solver_core as SC

img = np.asarray(fabio.open(os.path.join(ZOO, "Pilatus1M.edf")).data, float)
img[img >= 1e9] = 0.0
img[img < 0] = 0.0
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
    os.path.join(ZOO, "Pilatus1M.poni"))
canvas, s, _, _ = SC.to_canvas(img)
px_m = float(ai.detector.pixel1) * s
gt_D = ai.dist * 1000.0
print(f"canvas {canvas.shape} s={s} px_m={px_m*1000:.3f}mm GT_D={gt_D:.1f}mm")

qs = np.asarray(phase_qs("AgBh"), float)
print(f"AgB6 q线 {len(qs)} 条: {np.round(qs,4)[:12]}")

# 主解（名义 λ）拿中心与 D
r = SC.solve_auto(canvas, 1.0, px_m, cascade=True, auto_center=True)
ph, sol = r
print(f"主解: {ph} m={sol[3]} rms={sol[4]:.2f} ps={sol[6]:+.3f} "
      f"D={sol[2]*px_m*1000:.1f}mm c=({sol[0]:.1f},{sol[1]:.1f})")

# 峰表（看看 AgBh 到底几条环在量程内）
peaks, _, _, _ = SC.ring_radii(canvas, sol[0], sol[1])
print(f"峰表({len(peaks)}): " + " ".join(f"{p[0]:.1f}" for p in peaks))

# λ-D LSQ：从名义 λ 与主解 D 出发（±20%），dbg 看 σ
out = SC.refine_lambda_d(canvas, sol[0], sol[1], sol[2], 1.0, qs,
                         dbg=True, lam_frac=0.20)
print("refine_lambda_d:", out)
if out:
    lam_h, D_h, n, rms, sig = out
    print(f"→ λ̂={lam_h:.4f} (1.0326 期望, 偏差{abs(lam_h/1.0326-1)*100:.2f}%) "
          f"D@λ̂={D_h*px_m*1000:.2f}mm ΔD={D_h*px_m*1000-gt_D:+.2f}mm "
          f"σ(λ)/λ={sig}")

# 敏感度参考：固定 D=GT, λ=真值 vs ±0.5%，模型半径差多少 px
for lam in (1.0326, 1.0274, 1.0378):
    rm = SC._r_model(qs, lam, gt_D / px_m / 1000 * 1000 / (px_m * 1000) * px_m)
    rm = gt_D / 1000.0 / px_m * np.tan(
        2 * np.arcsin(np.clip(qs * lam / (4 * np.pi), 0, .9999)))
    vis = rm[(rm > 25) & (rm < 0.5 * max(canvas.shape))]
    print(f"λ={lam:.4f}: 可见环半径(px) {np.round(vis,1)}")
