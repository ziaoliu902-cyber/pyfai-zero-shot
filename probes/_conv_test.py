# -*- coding: utf-8 -*-
"""收敛性检验：卡壳三档 400 vs 2000 迭代。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
import fabio
import pyFAI

from real_refine_v6 import REAL, ROOT, load_px_nn, nn_coarse, to_nn_input_wide
from torchfai.refine import DifferentiableRefiner

m, dev, consts = load_px_nn(os.path.join(ROOT, "runs/px_v2/best.pt"))
ref6 = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus6M", wavelength_m=9.72386e-11,
                             dist_range=[0.1, 1.0])
for d in (200, 250, 400):
    img = fabio.open(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    gt = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni")).dist
    wide, k, extra = to_nn_input_wide(img)
    p_nn, _ = nn_coarse(m, dev, consts, wide, 0.972386, k)
    st = np.array([p_nn[0], p_nn[1], p_nn[2] + extra,
                   p_nn[3], p_nn[4], p_nn[5]])
    for iters in (400, 2000):
        r = ref6.refine(img, st, iters=iters, sigma_norm=None,
                        sigma_range=(0.0015, 0.02))
        fd = r["params"][0]
        print(f"{d}mm iters={iters}: {fd*1000:7.1f}mm "
              f"({(fd-gt)*1000:+7.2f}) loss={r['loss']:.2f}", flush=True)
