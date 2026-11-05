# -*- coding: utf-8 -*-
"""可微 q-map vs pyFAI 逐像素对拍（正确性闸门）。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.refine import DifferentiableRefiner
from torchfai.simulator import make_geometry, get_detector
from pyFAI.geometry import Geometry

ref = DifferentiableRefiner("AgBh", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train_v2"))
cfg, db = ref.cfg, ref.db

rng = np.random.default_rng(7)
worst = 0.0
worst_g = None
for k in range(12):
    # 覆盖全范围 + 极端倾斜组合
    g = dict(
        dist=float(10 ** rng.uniform(np.log10(0.1), np.log10(3.0))),
        poni1=float(rng.uniform(0.06, 0.94) * db.extent[0]),
        poni2=float(rng.uniform(0.06, 0.94) * db.extent[1]),
        rot1=float(rng.uniform(-0.07, 0.07)) * (1.8 if k in (2, 3) else 1.0),
        rot2=float(rng.uniform(-0.07, 0.07)) * (1.8 if k in (2, 4) else 1.0),
        rot3=float(rng.uniform(-0.17, 0.17)),
    )
    if k == 0:   # 加一例近距 + 一例远距的定点
        g.update(dist=0.1, rot1=0.0, rot2=0.0, rot3=0.0)
    if k == 1:
        g.update(dist=3.0, rot1=0.05, rot2=-0.05, rot3=0.17)

    pyf = Geometry(detector=db.pyfai_det, dist=g["dist"], poni1=g["poni1"],
                   poni2=g["poni2"], rot1=g["rot1"], rot2=g["rot2"], rot3=g["rot3"],
                   wavelength=cfg["wavelength"])
    q_ref = pyf.array_from_unit(shape=db.shape, typ="center", unit="q_A^-1", scale=True)
    q_ref = np.nan_to_num(q_ref, nan=100.0, posinf=100.0)

    pn = ref.phys_to_pn(np.array([g["dist"], g["poni1"], g["poni2"],
                                  g["rot1"], g["rot2"], g["rot3"]]), ref.consts)
    with torch.no_grad():
        q_t = ref.q_map(pn.to(ref.device)).cpu().numpy()

    d = np.abs(q_t - q_ref).max()
    if d > worst:
        worst, worst_g = d, g
    print(f"[{k:2d}] dist={g['dist']:.3f}m r1={g['rot1']:+.3f} r2={g['rot2']:+.3f} "
          f"r3={g['rot3']:+.3f}  max|Δq|={d:.3e} Å⁻¹")

print(f"\n最差: {worst:.3e} Å⁻¹  ({'PASS ✅' if worst < 1e-4 else 'FAIL ❌'})")
print("参考：典型环间距 Δq ≈ 0.1-1 Å⁻¹，像素级 q 分辨率 ~1e-3，故 1e-4 已是亚像素级安全")
