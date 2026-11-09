# -*- coding: utf-8 -*-
"""_b12_diag.py:B12② 失败诊断(2026-11-07)。

_b12_partial 全起点失败(rms 74-247px)。三步定位:
  a) 真值初值 → 拟合应回 rms~0.5px(模型/代码正确性)
  b) 中心±50px、D±2% 扰动 → 收敛半径
  c) 失败模式:简并方向(中心平移↔倾角)确认
"""
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


def synth(win_deg, nper=15, noise=0.5):
    pts = []
    w = np.deg2rad(win_deg)
    for j, tq in enumerate(TTH):
        ph = np.linspace(-w / 2, w / 2, nper) + rng.normal(0, 0.02, nper)
        r = D_PX * np.tan(tq + ROT * np.cos(ph)) + rng.normal(0, noise, nper)
        pts.append((CX + r * np.cos(ph), CY + r * np.sin(ph),
                    np.full(nper, j)))
    return (np.concatenate([p[0] for p in pts]),
            np.concatenate([p[1] for p in pts]),
            np.concatenate([p[2] for p in pts]))


def fit(p0, x, y, jidx, soft=True):
    def resid(p):
        Dp, cx, cy, a, b = p
        ph = np.arctan2(y - cy, x - cx)
        r = np.hypot(x - cx, y - cy)
        return r - Dp * np.tan(TTH[jidx] + a * np.cos(ph) + b * np.sin(ph))
    kw = dict(loss="soft_l1", f_scale=2.0) if soft else {}
    return least_squares(resid, p0, max_nfev=8000, **kw)


print("== a) 真值初值(模型正确性) ==")
for tag, win in (("279°", 279.0), ("120°", 120.0)):
    x, y, jidx = synth(win)
    f = fit([D_PX, CX, CY, ROT, 0.0], x, y, jidx)
    rms = np.sqrt(np.mean(f.fun ** 2))
    print(f"  {tag}: rms={rms:.2f}px  D={f.x[0]*PX*1e3:.3f}mm "
          f"(Δ{f.x[0]*PX*1e3-200.19:+.3f})  a={f.x[3]:+.3f}(真 {ROT:+.3f})")

print("\n== b) 扰动扫描(soft_l1 vs 纯 L2) ==")
x, y, jidx = synth(120.0)
for dsc, dp in (("D±2% 中心±50", (0.02, 50)),
                ("D±5% 中心±150", (0.05, 150)),
                ("D±10% 中心±300", (0.10, 300))):
    for soft in (True, False):
        rg = np.random.default_rng(9)
        oks = 0
        for k in range(6):
            p0 = [D_PX * rg.uniform(1 - dp[0], 1 + dp[0]),
                  CX + rg.uniform(-dp[1], dp[1]),
                  CY + rg.uniform(-dp[1], dp[1]),
                  ROT + rg.uniform(-0.2, 0.2), rg.uniform(-0.2, 0.2)]
            f = fit(p0, x, y, jidx, soft)
            if abs(f.x[0] * PX * 1e3 - 200.19) < 2.0:
                oks += 1
        print(f"  {dsc:16} {'soft_l1' if soft else 'L2':7} 收敛 {oks}/6")

print("\n== c) 简并确认:真解处把中心 +200px 沿 x,补偿 (a) 后残差? ==")
p = [D_PX, CX + 200, CY, ROT, 0.0]
ph = np.arctan2(y - (CY), x - (CX + 200))
r = np.hypot(x - (CX + 200), y - CY)
r_t = D_PX * np.tan(TTH[jidx] + ROT * np.cos(np.arctan2(y - CY, x - CX)))
print(f"  中心+200px 不补偿: 残差rms={np.sqrt(np.mean((r-r_t)**2)):.1f}px")
# 小角近似:D·a 吸收 Δcx:resid ≈ Δcx·cosφ → 拟合 a' = a − Δcx/D
f = fit(p, x, y, jidx, soft=False)
print(f"  中心+200px 初值放开 5 参: 收敛到 cx={f.x[1]:.0f} "
      f"D={f.x[0]*PX*1e3:.2f}mm a={f.x[3]:+.3f} "
      f"rms={np.sqrt(np.mean(f.fun**2)):.2f}px")
