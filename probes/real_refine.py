# -*- coding: utf-8 -*-
"""NC 对拍主考卷：MX_ceria 七档 × 物理精修（真实 λ/探测器 override）。

起点两轨：
  A 名义几何：dist=档距标称值, poni=探测器中心, rot=0（模拟"只知道大概装哪"）
  B 真值起点：官方 poni（物理层自洽性对照——真值应是不动点）
裁判：官方多帧标定 poni（150档 7 次标定抖动 ±0.07mm）。
用法：python probes/real_refine.py [--smoke]  （--smoke 只跑 ceria_300 名义轨）
"""
import argparse
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio
import pyFAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")
LAM = 9.72386e-11                    # ceria 系列波长（poni 里写明，已交叉验证）
DISTS = (150, 200, 250, 300, 350, 400, 450)


def gt_poni(d):
    """官方 poni → 六参数真值 (m, m, m, rad, rad, rad)。"""
    p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
    return np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])


def main():
    from torchfai.refine import DifferentiableRefiner

    a = argparse.ArgumentParser()
    a.add_argument("--smoke", action="store_true")
    args = a.parse_args()

    ref = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3"),
                                detector="Pilatus6M", wavelength_m=LAM,
                                dist_range=[0.1, 1.0])
    Hm, Wm = ref.db.extent
    px = ref.db.pixel
    print(f"Pilatus6M {ref.db.shape} px={px*1e6:.0f}μm λ={ref.lam_A:.4f}Å "
          f"dead={ref.db.dead.mean()*100:.1f}%")

    dists = (300,) if args.smoke else DISTS
    print(f"\n{'档':>4s} {'轨':4s} {'起点Δmm':>8s} {'终点Δmm':>8s} {'poni1Δpx':>9s}"
          f" {'poni2Δpx':>9s} {'rot1Δmrad':>10s} {'loss_init':>10s} {'loss':>10s}"
          f" {'iters':>6s} {'sec':>5s}")
    rows = []
    for d in dists:
        img = fabio.open(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        gt = gt_poni(d)
        starts = {
            "名义": np.array([d / 1000.0, Hm / 2, Wm / 2, 0.0, 0.0, 0.0]),
            "真值": gt,
        }
        for tag, st in starts.items():
            t0 = time.time()
            r = ref.refine(img, st, iters=400, sigma_norm=None, trust_k=1.0)
            e0 = (st[0] - gt[0]) * 1000
            e1 = (r["params"][0] - gt[0]) * 1000
            row = (d, tag, e0, e1,
                   (r["params"][1] - gt[1]) / px, (r["params"][2] - gt[2]) / px,
                   (r["params"][3] - gt[3]) * 1000,
                   r["loss_init"], r["loss"], r["iters"], time.time() - t0)
            rows.append(row)
            print(f"{d:4d} {tag:4s} {e0:8.1f} {e1:8.2f} {row[4]:9.2f} "
                  f"{row[5]:9.2f} {row[6]:10.2f} {r['loss_init']:10.3e} "
                  f"{r['loss']:10.3e} {r['iters']:6d} {row[10]:5.0f}", flush=True)

    if not args.smoke:
        fn = np.array([r[3] for r in rows if r[1] == "名义"])
        print(f"\n名义轨 dist 终误差: " +
              " ".join(f"{v:+.2f}" for v in fn) + f"  (中位 {np.median(np.abs(fn)):.2f}mm)")


if __name__ == "__main__":
    main()
