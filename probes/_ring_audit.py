# -*- coding: utf-8 -*-
"""_ring_audit.py：250 档逐环对账——检测峰半径 vs GT 模型半径（px）。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI
import torch

from real_refine_v6 import load_px_nn, to_nn_input_wide
from ring_fit import (HKLS, LAM, MIN_R, NB, PXEFF, Q_HKL, nn_coarse,
                      ring_radii, r_model)

d = 450
img = fabio.open(os.path.join(
    ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
p = pyFAI.load(os.path.join(
    ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
p6, wide, k = nn_coarse(img)
Hm, Wm = wide.shape
print(f"canvas {Hm}x{Wm}  k={k:.3f}  NN dist={p6[0]*1000:.1f}mm GT={p.dist*1000:.1f}mm")

# GT 中心（canvas 坐标）
cy_gt = p.poni1 / PXEFF * (Hm / 521.0)
cx_gt = p.poni2 / PXEFF * (Wm / 490.0)
D_gt = p.dist / PXEFF * (Hm / 521.0)
print(f"GT 中心=({cx_gt:.1f},{cy_gt:.1f}) D_px={D_gt:.1f}")

peaks = ring_radii(wide, cx_gt, cy_gt)
print(f"\n检测峰（GT中心，共{len(peaks)}个）：")
for rp, h in peaks:
    print(f"  r={rp:7.2f}px  峰高={h:9.1f}")

print(f"\n模型半径（GT 几何）：")
for (h_, k_, l_), q in zip(HKLS, Q_HKL):
    rm = r_model(np.array([q]), cx_gt, cy_gt, D_gt)[0]
    near = min(peaks, key=lambda t: abs(t[0] - rm)) if peaks else (np.nan, 0)
    tag = f"→ 检测 {near[0]:7.2f} (Δ{near[0]-rm:+6.2f}px)" if peaks else ""
    vis = "域内" if rm < np.hypot(cx_gt, cy_gt) * 1.4 else "可能出域"
    print(f"  ({h_}{k_}{l_}) q={q:.3f} r={rm:7.2f}px  {tag}  {vis}")
