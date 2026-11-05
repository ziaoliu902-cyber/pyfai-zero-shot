# -*- coding: utf-8 -*-
"""束心审判：参考 poni 的 (poni1,poni2) 是否与图像自洽？

似然裁决：poni 钉死三处（GT / 直测质心 / 剖面锐度），dist 自由精修，
比最终损失。若 GT 处损失显著更差 → 参考 poni 的束心也不自洽
（与 λ 元数据 3.2% 案并案）。
"""
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from torchfai.refine import DifferentiableRefiner
from real_refine_v6 import REAL, load_geom
from _agbh_final import ring_centroid, PX
from _agbh_center_fit import ring_center_profile

img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])

cy_d, cx_d, _ = ring_centroid(img.astype(np.float64))
cy_p, cx_p, _ = ring_center_profile(img.astype(np.float64))

ref = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3x"),
                            detector="Pilatus_1M", wavelength_m=1.0e-10,
                            dist_range=[0.1, 3.0])
PIN = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

print(f"GT poni px = ({gt[1]/PX:.1f},{gt[2]/PX:.1f})   "
      f"直测=({cy_d:.1f},{cx_d:.1f})   剖面=({cy_p:.1f},{cx_p:.1f})\n")
for tag, cy, cx in (("GT 束心", gt[1] / PX, gt[2] / PX),
                    ("直测质心", cy_d, cx_d),
                    ("剖面锐度", cy_p, cx_p),
                    ("GT/直测中点", (gt[1] / PX + cy_d) / 2,
                     (gt[2] / PX + cx_d) / 2)):
    st = np.array([1.642, cy * PX, cx * PX, gt[3], gt[4], gt[5]])
    out = ref.refine(img, st, iters=300, lr_geo=8e-3, lr_mult=PIN,
                     sigma_range=(0.0015, 0.6))
    d, l = out["params"][0], out["loss"]
    print(f"{tag:10s} ({cy:6.1f},{cx:6.1f})  dist={d*1000:7.1f}mm"
          f"({(d-gt[0])*1000:+7.1f})  loss={l:10.1f}", flush=True)
