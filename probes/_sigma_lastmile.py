# -*- coding: utf-8 -*-
"""最后一公里：卡壳档位放宽 sigma_range 复测（A:200/250/400 + AgBh）。

假盆地产于 sigma_range=(0.0015,0.02) 的瘦环约束与真实胖环不匹配
（alumina 放宽到 0.6 后 +941→+1.67mm 的同款药方）。
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabio
import pyFAI

from real_refine_v6 import (LAM_CERIA, REAL, ROOT, load_px_nn, nn_coarse,
                            to_nn_input_wide)
from torchfai.refine import DifferentiableRefiner

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "runs/px_v2/best.pt"))
args = ap.parse_args()
m, dev, consts = load_px_nn(args.model)

SIGMAS = ((0.0015, 0.02), (0.0015, 0.06), (0.0015, 0.15), (0.0015, 0.4))
ref6 = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus6M", wavelength_m=LAM_CERIA,
                             dist_range=[0.1, 1.0])

for d in (200, 250, 400):
    img = fabio.open(os.path.join(
        REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    wide, k, extra = to_nn_input_wide(img)
    p_nn, _ = nn_coarse(m, dev, consts, wide, 0.972386, k)
    st = np.array([p_nn[0], p_nn[1], p_nn[2] + extra,
                   p_nn[3], p_nn[4], p_nn[5]])
    line = f"[ceriaW {d}mm]"
    for sr in SIGMAS:
        r = ref6.refine(img, st, iters=400, sigma_norm=None, sigma_range=sr)
        line += (f"  σ≤{sr[1]:g}:{(r['params'][0]-gt[0])*1000:+7.2f}mm"
                 f"(σ̄={np.mean(r.get('sigmas', [np.nan])):.3f})")
    print(line, flush=True)

print("-- AgBh@1M --")
ref2 = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus_1M", wavelength_m=1.000e-10,
                             dist_range=[0.1, 3.0])
img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
p = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))
gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
H2, W2 = ref2.db.extent
from real_refine_v5 import to_nn_input_phys
from real_refine_v6 import match_gain
p_nn, _ = nn_coarse(m, dev, consts,
                    match_gain(to_nn_input_phys(img)) * 0.5, 1.0, 1.0)
st = np.array([p_nn[0], gt[1], gt[2], 0.0, 0.0, 0.0])
for sr in SIGMAS:
    r = ref2.refine(img, st, iters=400, sigma_norm=None, sigma_range=sr)
    print(f"  σ≤{sr[1]:g}: {(r['params'][0]-gt[0])*1000:+7.2f}mm", flush=True)
