# -*- coding: utf-8 -*-
"""AgBh +50mm 地板的三连敏感度：λ 标度 / rot 松绑 / 饱和像素。

若某因素消掉大半残差 → 找到地板来源，且那不是模型误差而是
物理量未对齐（论文里叫系统误差分解）。
"""
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from torchfai.refine import DifferentiableRefiner
from real_refine_v6 import REAL, load_geom
from _agbh_final import ring_centroid

PX = 172e-6


def run(tag, lam_m, rot_free, include_sat):
    img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
    p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    cy, cx, _ = ring_centroid(img.astype(np.float64))
    ref = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3x"),
                                detector="Pilatus_1M", wavelength_m=lam_m,
                                dist_range=[0.1, 3.0])
    lm = (1.0, 0.05, 0.05, 0.02 if rot_free else 0.0,
          0.02 if rot_free else 0.0, 0.02 if rot_free else 0.0)
    st = np.array([1.642, cy * PX, cx * PX, 0, 0, 0])
    out = ref.refine(img, st, iters=300, lr_geo=8e-3, lr_mult=lm,
                     sigma_range=(0.0015, 0.6), include_sat=include_sat)
    q = out["params"]
    print(f"{tag:22s} distΔ={(q[0]-gt[0])*1000:+7.1f}mm  "
          f"loss={out['loss']:9.1f}  "
          f"rot=({q[3]:+.4f},{q[4]:+.4f},{q[5]:+.4f})", flush=True)
    return out["loss"], q


if __name__ == "__main__":
    LAM0 = 1.000e-10
    run("基线 λ=1.00 rot钉 sat排除", LAM0, False, False)
    run("λ=0.97", 0.97e-10, False, False)
    run("λ=1.03", 1.03e-10, False, False)
    run("λ=1.05", 1.05e-10, False, False)
    run("rot 松绑", LAM0, True, False)
    run("饱和像素纳入", LAM0, False, True)
    run("λ=1.03 + rot松", 1.03e-10, True, False)
