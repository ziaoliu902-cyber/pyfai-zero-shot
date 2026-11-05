# -*- coding: utf-8 -*-
"""宽视域可见环数冒烟（gen_lamp 改造决策依据，纯几何计算）。

问题：ceria@λ0.9724Å 在各距离下，训练视域 vs 6M 视域各能看见几条环？
（生成器规则：可见环<2 剔除 → 可见环数 = 覆盖洞的边界）
环可见判据：环半径 r = dist·tan(2θ) ≤ 探测器半对角（poni 居中最优情形）。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.phases import phase_qi

qs, Is = phase_qi("CeO2")
LAM = 0.9724
FOVS = {
    "训练视域 0.179x0.169": (0.179, 0.169),
    "6M 视域 0.435x0.424": (0.435, 0.424),
}
print(f"CeO2 前 6 环 q = {np.round(qs[:6], 3)}  λ={LAM} Å\n")
print(f"{'dist':>6}", *[f"{n:>22}" for n in FOVS])
for d in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.60, 1.0, 1.58):
    line = f"{d*1000:5.0f}m"
    for name, (h, w) in FOVS.items():
        half = np.sqrt(h * h + w * w) / 2
        r = d * np.tan(2 * np.arcsin(np.minimum(qs * LAM / (4 * np.pi), 1.0)))
        n_vis = int((r < half).sum())
        line += f"{n_vis:>15}环(r1={r[0]*1000:5.0f}mm)"
    print(line)

print("""
判读：
- 生成器剔除线 = 2 环 → 训练视域在 ~300mm 起 ceria@λ0.97 跌破 2 环（覆盖洞）
- 6M 视域全程 ≥3 环 → 宽视域渲染物理上能填洞
- AgBh 对照（低 q 环）：不受视域限制（agbh_far 有效的原因）""")
qs2, _ = phase_qi("AgBh")
r2 = 1.58 * np.tan(2 * np.arcsin(qs2 * 1.0 / (4 * np.pi)))
print(f"AgBh@λ1.0@1583mm 前 3 环半径 = {np.round(r2[:3]*1000, 1)}mm"
      f"  → 训练视域可见 {int((r2 < np.sqrt(0.179**2+0.169**2)/2).sum())} 环+")
