# -*- coding: utf-8 -*-
"""_t_joint8b.py:盲解 −2.1mm 系统偏溯源(2026-11-08)。

win 扫描已排除峰窗;GT λ=0.7084811 vs 锁相值 0.7084571(0.003%)排除 λ。
对照:第 1 帧三种配置的 D̂:
  a) 锁相 λ(已知 Δ−0.22)
  b) λ 联立(λ̂ 自由)→ 看 λ̂ 与 D̂(若 D̂ 回 116.8 → 误差被 λ̂ 吸收,
     即锁相 λ 与链内有效波长有系统差;若 D̂ 仍 −2 → 链本身偏)
  c) 锁相 GT λ(0.7084811)→ 差异应微小
"""
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc

PX1 = 103.358e-6
GT = 116.860
img = fabio.open(ROOT + r"\data\real\alumina\al2o3_0000.edf").data
img = np.where(np.isfinite(img), img, 0.0).astype(np.float64)

print("帧 al2o3_0000 三配置对照:")
for tag, lam in (("锁相 12.398/17.5", 12.398 / 17.5),
                 ("锁相 GT λ", 0.7084811),
                 ("λ 联立", None)):
    r = sc.solve_image(img, PX1, lam=lam, phases=["Al2O3"])
    if r is None:
        print(f"  [{tag}] 无解")
        continue
    D = r["sol"][2] * r["s"] * PX1 * 1e3
    lam_out = r.get("lam")
    print(f"  [{tag}] D={D:.3f}mm (Δ{D-GT:+.3f}) "
          f"λ_used={lam_out:.7f} m={r['sol'][3]} rms={r['sol'][4]:.3f}px "
          f"s={r['s']}", flush=True)
