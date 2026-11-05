# -*- coding: utf-8 -*-
"""自由 poni 精修：终点 poni 到底停在哪（此前从未打印）。"""
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probes"))
from torchfai.refine import DifferentiableRefiner
from real_refine_v6 import REAL, load_geom
from _agbh_final import ring_centroid, PX

img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
cy, cx, _ = ring_centroid(img.astype(np.float64))
ref = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3x"),
                            detector="Pilatus_1M", wavelength_m=1.0e-10,
                            dist_range=[0.1, 3.0])
st = np.array([1.642, cy * PX, cx * PX, 0, 0, 0])
out = ref.refine(img, st, iters=600, lr_geo=8e-3,
                 lr_mult=(1.0, 0.05, 0.05, 0.0, 0.0, 0.0),
                 sigma_range=(0.0015, 0.6))
q = out["params"]
print(f"起点 poni=({cy:.1f},{cx:.1f})px")
print(f"终点 poni=({q[1]/PX:.1f},{q[2]/PX:.1f})px  "
      f"GT=({gt[1]/PX:.1f},{gt[2]/PX:.1f})px")
print(f"终点 dist={q[0]*1000:.1f}mm ({(q[0]-gt[0])*1000:+.1f})  "
      f"loss={out['loss']:.1f}")
