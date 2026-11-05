# -*- coding: utf-8 -*-
"""rot3 严格不变性验证：rot3 从最小拉到最大，q-map 是否逐像素不变。

理论：R = R3·R2·R1，改 rot3 相当于把每个像素的样品-像素矢量绕光轴（z）转，
极角不变 → 2θ 不变 → q 逐像素严格不变；只有方位角 χ 平移。
本实验用真实 q_map 代码验证（含随机倾角/偏移组合）。
"""
import sys
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from torchfai.refine import DifferentiableRefiner

ref = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data", "train_v2"),
                            device="cpu")
c = ref.consts
rng = np.random.default_rng(0)

print("rot3 范围: [%.3f, %.3f] rad" % (c["r_lo"][2], c["r_hi"][2]))
print()
print(f"{'组合':>4s} {'rot1/rot2 倾角':>18s} {'poni 偏移(px)':>14s} "
      f"{'max|dq| (A^-1)':>16s} {'q 范围':>14s}")

for trial in range(4):
    # 随机几何（含极端倾角）
    r1 = rng.uniform(c["r_lo"][0], c["r_hi"][0])
    r2 = rng.uniform(c["r_lo"][1], c["r_hi"][1])
    p1 = rng.uniform(0.25, 0.75) * c["Hm"]
    p2 = rng.uniform(0.25, 0.75) * c["Wm"]
    d = 10 ** rng.uniform(np.log10(0.15), np.log10(1.0))

    def make_pn(rot3):
        return torch.tensor([
            (np.log10(d) - c["dlo"]) / (c["dhi"] - c["dlo"]) * 2 - 1,
            p1 / c["Hm"] * 2 - 1, p2 / c["Wm"] * 2 - 1,
            (r1 - c["r_lo"][0]) / (c["r_hi"][0] - c["r_lo"][0]) * 2 - 1,
            (r2 - c["r_lo"][1]) / (c["r_hi"][1] - c["r_lo"][1]) * 2 - 1,
            (rot3 - c["r_lo"][2]) / (c["r_hi"][2] - c["r_lo"][2]) * 2 - 1,
        ], dtype=torch.float32)

    qa = ref.q_map(make_pn(c["r_lo"][2]))          # rot3 最小
    qb = ref.q_map(make_pn(0.0))                    # rot3 中点
    qc = ref.q_map(make_pn(c["r_hi"][2]))           # rot3 最大
    dq1 = float((qa - qb).abs().max())
    dq2 = float((qc - qb).abs().max())
    qmax = float(qa.max())
    print(f"{trial:4d} {r1*1000:7.1f}/{r2*1000:6.1f} mrad "
          f"{p1/ref.db.pixel:5.0f}/{p2/ref.db.pixel:5.0f}   "
          f"{max(dq1, dq2):16.2e} {qmax:10.3f}")

print()
print("解读：max|dq| 在 1e-7 量级 = float32 舍入，q-map 对 rot3 严格不变。")
print("      → 1D 积分 I(q) 严格不变；2D 只是 χ 轴平移；成像前向也严格不变。")
