# -*- coding: utf-8 -*-
"""AgBh 终局加时赛：dist + poni 小步长联合精修（从直测环心起，不用 GT）。

前一战：poni 钉直测 (256,180)（GT (194,240) 偏 +61/-60px）→ dist +58.8mm。
假设：残差主要是 poni 偏差向 dist 的耦合。联合精修 poni 小步长（×0.05）
让两者互相校准；若跑飞（>±120px）则弃，保 +58.8 保守值。
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

PX = 172e-6
from _agbh_final import ring_centroid  # 复用迭代环心


def main():
    img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
    p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    cy, cx, _ = ring_centroid(img.astype(np.float64))

    ref = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3x"),
                                detector="Pilatus_1M",
                                wavelength_m=1.000e-10,
                                dist_range=[0.1, 3.0])
    best = None
    for d_start in (1.487, 1.642, 1.75):
        st = np.array([d_start, cy * PX, cx * PX, 0, 0, 0])
        out = ref.refine(img, st, iters=300, lr_geo=8e-3,
                         lr_mult=(1.0, 0.05, 0.05, 0.0, 0.0, 0.0),
                         sigma_range=(0.0015, 0.6))
        q = out["params"]
        dy = (q[1] - gt[1]) / PX
        dx = (q[2] - gt[2]) / PX
        ed = (q[0] - gt[0]) * 1000
        print(f"start {d_start:.3f} → dist {q[0]*1000:.1f}mm ({ed:+.1f})  "
              f"poniΔ=({dy:+.0f},{dx:+.0f})px  loss={out['loss']:.1f}",
              flush=True)
        if best is None or out["loss"] < best[0]:
            best = (out["loss"], q, ed, dy, dx)
    print(f"\n最优：distΔ={best[2]:+.1f}mm poniΔ=({best[3]:+.0f},{best[4]:+.0f})px"
          f"  loss={best[0]:.1f}")
    if abs(best[3]) > 120 or abs(best[4]) > 120:
        print("poni 跑飞 → 弃用，保守值 +58.8mm")


if __name__ == "__main__":
    main()
