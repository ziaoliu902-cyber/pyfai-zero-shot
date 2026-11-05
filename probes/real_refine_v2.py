# -*- coding: utf-8 -*-
"""ceria 对拍 v2：距离扫描多起点（dist 网格 × poni 中心）纯似然择优。

v1 结论（probes/real_refine.py）：名义起点 7/7 dist 收敛（中位 1.06mm）但
250 档 dist 假谷 +3.65mm；poni↔rot 简并方向全部卡浅谷（rot1 恒 -5~-7mrad）。
v2：每档跑 dist 扫描起点（0.15~0.60 对数 6 点 + 名义档距），纯似然选最深。
"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio
import pyFAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from torchfai.refine import DifferentiableRefiner

LAM = 9.72386e-11
DISTS = (150, 200, 250, 300, 350, 400, 450)
SCAN = (0.15, 0.20, 0.27, 0.36, 0.48, 0.60)


def main():
    ref = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3"),
                                detector="Pilatus6M", wavelength_m=LAM,
                                dist_range=[0.1, 1.0])
    Hm, Wm = ref.db.extent
    px = ref.db.pixel
    print(f"{'档':>4s}{'最优起点':>9s}{'loss_init':>10s}{'loss':>10s}{'distΔmm':>9s}"
          f"{'poni1Δpx':>9s}{'poni2Δpx':>9s}{'rot1Δmrad':>10s}{'sec':>5s}")
    errs = []
    for d in DISTS:
        img = fabio.open(os.path.join(
            ROOT, "data/real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(
            ROOT, "data/real", f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        t0 = time.time()
        best = None
        for sd in sorted(set(SCAN + (d / 1000.0,))):
            st = np.array([sd, Hm / 2, Wm / 2, 0.0, 0.0, 0.0])
            r = ref.refine(img, st, iters=400, sigma_norm=None)
            if best is None or r["loss"] < best[1]["loss"] - 1e-9:
                best = (sd, r)
        sd, r = best
        ed = (r["params"][0] - gt[0]) * 1000
        errs.append(ed)
        print(f"{d:4d}{sd:9.3f}{r['loss_init']:10.2f}{r['loss']:10.2f}{ed:9.2f}"
              f"{(r['params'][1]-gt[1])/px:9.1f}{(r['params'][2]-gt[2])/px:9.1f}"
              f"{(r['params'][3]-gt[3])*1000:10.2f}{time.time()-t0:5.0f}",
              flush=True)
    errs = np.array(errs)
    print(f"\ndist 终误差: " + " ".join(f"{v:+.2f}" for v in errs)
          + f"\n中位 {np.median(np.abs(errs)):.2f}mm  最大 {np.abs(errs).max():.2f}mm")


if __name__ == "__main__":
    main()
