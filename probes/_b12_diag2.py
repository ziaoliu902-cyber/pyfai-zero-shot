# -*- coding: utf-8 -*-
"""_b12_diag2.py:真值初值全参数+分环残差(2026-11-07)。"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from torchfai.phases import phase_qs

PX = 172e-6
LAM = 0.972386
D_PX = 200.19e-3 / PX
Q = phase_qs("CeO2")[:9]
TTH = 2 * np.arcsin(np.clip(Q * LAM / (4 * np.pi), 0, 0.999))
CX, CY = -300.0, 1263.5
ROT = np.deg2rad(-29.0)
rng = np.random.default_rng(21)

w = np.deg2rad(279.0)
xs, ys, js = [], [], []
for j, tq in enumerate(TTH):
    ph = np.linspace(-w / 2, w / 2, 15) + rng.normal(0, 0.02, 15)
    r = D_PX * np.tan(tq + ROT * np.cos(ph)) + rng.normal(0, 0.5, 15)
    xs.append(CX + r * np.cos(ph)); ys.append(CY + r * np.sin(ph))
    js.append(np.full(15, j))
x = np.concatenate(xs); y = np.concatenate(ys); jidx = np.concatenate(js)

# 真值处直接算残差(不拟合!)
ph = np.arctan2(y - CY, x - CX)
r = np.hypot(x - CX, y - CY)
r_model = D_PX * np.tan(TTH[jidx] + ROT * np.cos(ph))
res0 = r - r_model
print(f"真值处残差(不拟合): rms={np.sqrt(np.mean(res0**2)):.3f}px "
      f"max={np.abs(res0).max():.3f}px")
for j in range(9):
    m = jidx == j
    print(f"  环{j} 2θ={np.degrees(TTH[j]):6.2f}°  "
          f"rms={np.sqrt(np.mean(res0[m]**2)):7.3f}px  "
          f"r范围 {r[m].min():.0f}~{r[m].max():.0f}px")
