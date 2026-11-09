# -*- coding: utf-8 -*-
"""_t_joint8e.py:mean-frame 帧数效应——52 帧全平均(2026-11-08)。

joint8d:8 帧平均种子 Δ+2.6(每帧 MAD 已恢复 0.085)。历史 116.79
(−0.07)是 52 帧 mean-frame。对照:52 帧平均锁相盲解;若 Δ≤0.3 →
弱衍射平均帧数是关键,生产链 mean=全部帧 ✓。
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
LAM = 12.398 / 17.5
GT = 116.860
AL = os.path.join(ROOT, "data", "real", "alumina")
frames = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                and f.endswith(".edf"))
print(f"52 帧全平均 mean-frame(锁相 λ={LAM:.7f})…", flush=True)
acc = None
for f in frames:
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    acc = img.copy() if acc is None else acc + img
mean_img = acc / len(frames)
r = sc.solve_image(mean_img, PX1, lam=LAM, phases=["Al2O3"])
assert r is not None, "无解"
s = r["s"]
D = r["sol"][2] * s * PX1 * 1e3
print(f"  52 帧 mean 种子:D={D:.3f}mm (Δ{D-GT:+.3f}) m={r['sol'][3]} "
      f"rms={r['sol'][4]:.3f}px s={s}")
print("判决:" + ("✓ Δ≤0.3(帧数是关键)" if abs(D - GT) <= 0.3
                  else "✗ 52 帧 mean 仍偏(查 λ 联立对照)"))
