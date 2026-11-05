# -*- coding: utf-8 -*-
"""远距翻案 B 案：px_v2 裁切粗解（k=1）× 中心起点网格精修，五档远距。

发现：ceria_200 同帧 B 案精修 +0.09mm 而 A 案卡 -85mm → 假盆地是宽视域
精修路径特有。px_v2 的 far@k=1 重放让裁切粗解复活（+25/+11.7），
若五档全 <1mm 则远距治愈只需裁切路径。
"""
import argparse
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabio
import pyFAI

from real_refine_v5 import to_nn_input_phys
from real_refine_v6 import GRID_MM, LAM_CERIA, REAL, ROOT, load_px_nn, nn_coarse
from torchfai.refine import DifferentiableRefiner

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "runs/px_v2/best.pt"))
args = ap.parse_args()
m, dev, consts = load_px_nn(args.model)

refc = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus6M", wavelength_m=LAM_CERIA,
                             dist_range=[0.1, 1.0])
Hm6, Wm6 = refc.db.extent
out_all = {}
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    p_nn, _ = nn_coarse(m, dev, consts, to_nn_input_phys(img), 0.972386, 1.0)
    starts = {f"nn{dm:+.0f}": np.array(
        [np.clip(p_nn[0] + dm * 1e-3, 0.1, 1.0), Hm6 / 2, Wm6 / 2, 0, 0, 0])
        for dm in GRID_MM}
    out = {}
    for tag, st in starts.items():
        r = refc.refine(img, st, iters=400, sigma_norm=None,
                        sigma_range=(0.0015, 0.02))
        out[tag] = (r["loss"], (r["params"][0] - gt[0]) * 1000)
    best = min(out, key=lambda t: out[t][0])
    out_all[d] = out[best][1]
    print(f"[crop ceria {d}mm] 粗解={(p_nn[0]-gt[0])*1000:+.1f} → "
          f"精修 {out[best][1]:+.2f}mm [{best}] "
          f"(loss={out[best][0]:.1f})", flush=True)
print("\n== 汇总(mm):", " ".join(f"{d}:{v:+.2f}" for d, v in out_all.items()))
