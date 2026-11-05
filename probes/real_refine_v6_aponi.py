# -*- coding: utf-8 -*-
"""v6 A 案补丁精修：起点带网络全几何（dist 网格 × NN poni/rots）。

背景：v6 考卷 A 案精修起点硬编码画布中心 poni（网络 poni 还是垃圾年代
的遗产），丢掉网络 ±9px 的 poni 估计 → 精修掉进距离-相位假盆地
（200mm 越修越歪到 -85mm）。本脚本只重跑 A 案七档，其余考题看主考卷。

用法：python -u -X utf8 probes/real_refine_v6_aponi.py --model runs/px_v2/best.pt
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabio
import pyFAI

from real_refine_v6 import (GRID_MM, LAM_CERIA, REAL, ROOT, load_px_nn,
                            nn_coarse, to_nn_input_wide)
from torchfai.refine import DifferentiableRefiner

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "runs/px_v2/best.pt"))
args = ap.parse_args()
m, dev, consts = load_px_nn(args.model)

ref6 = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus6M", wavelength_m=LAM_CERIA,
                             dist_range=[0.1, 1.0])
rows = []
A = {}
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    wide, k, extra = to_nn_input_wide(img)
    p_nn, sig = nn_coarse(m, dev, consts, wide, 0.972386, k)
    poni2_6m = p_nn[2] + extra                    # canvas 列偏移折回 6M 原生系
    poni_err = ((p_nn[1] - gt[1]) / 172e-6, (poni2_6m - gt[2]) / 172e-6)
    starts = {f"nn{dm:+.0f}": np.array(
        [np.clip(p_nn[0] + dm * 1e-3, 0.1, 1.0),
         p_nn[1], poni2_6m, p_nn[3], p_nn[4], p_nn[5]])
        for dm in GRID_MM}
    out = {}
    for tag, st in starts.items():
        t0 = time.time()
        r = ref6.refine(img, st, iters=400, sigma_norm=None,
                        sigma_range=(0.0015, 0.02))
        out[tag] = dict(loss=r["loss"], ed=(r["params"][0] - gt[0]) * 1000)
        rows.append(["aponi", f"ceriaW {d}mm", tag, f"{st[0]*1000:.1f}",
                     f"{r['loss']:.2f}", f"{out[tag]['ed']:+.3f}"])
    best = min(out.values(), key=lambda r: r["loss"])
    A[d] = ((p_nn[0] - gt[0]) * 1000, best["ed"])
    print(f"[ceriaW {d}mm] 粗解={p_nn[0]*1000:.1f}({A[d][0]:+.1f}) "
          f"poniΔ=({poni_err[0]:+.0f},{poni_err[1]:+.0f})px "
          f"→ 网格最优(带NN几何) {best['ed']:+.2f}mm "
          f"[min-start {min(out, key=lambda t: out[t]['loss'])}]",
          flush=True)

print("\n== A 汇总（粗解/精修, mm）:",
      " ".join(f"{d}:{A[d][0]:+.0f}/{A[d][1]:+.2f}" for d in A), flush=True)
ts = time.strftime("%Y%m%d_%H%M%S")
out_csv = os.path.join(ROOT, "probes", "results",
                       f"real_refine_v6_aponi_{ts}.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "case", "start", "start_dist_mm", "loss", "dist_err_mm"])
    w.writerows(rows)
print(f"落盘 -> {out_csv}")
