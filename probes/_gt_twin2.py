# -*- coding: utf-8 -*-
"""GT 孪生 v2（正确姿势）：db=get_scaled_detector(k)，pixel_scale 只做标签。
钉 GT 几何 → 若孪生准而真实帧偏 → 图像内容差；孪生也偏 → k=2.425 缝隙。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
from real_refine_v6 import load_px_nn
from torchfai.config import SimConfig
from torchfai.dataset import norm_to_param
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

m, dev, consts = load_px_nn(r"runs\v3far4\last.pt")
m.eval()
LAM = 0.972386


def coarse_dist(canvas, k):
    x = torch.from_numpy(
        np.log1p(canvas[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([LAM], device=dev),
                  torch.tensor([k], device=dev))
    return float(norm_to_param(mu.double().cpu().numpy(), consts, k)[0, 0])


ph_i = PHASE_NAMES.index("CeO2")
print("孪生（正确 db，钉 GT 几何，k=2.425 缝隙 vs 2.5 档）:")
for k in (2.425, 2.5):
    db = get_scaled_detector("Pilatus_1M", 2, k)
    Hm, Wm = db.extent
    rng = np.random.default_rng(7)
    for D in (0.2492, 0.3995):
        biases = []
        for t in range(3):
            g = dict(dist=D, poni1=Hm * 0.495, poni2=Wm * 0.502,
                     rot1=0.0, rot2=0.0, rot3=0.0)
            cfg = SimConfig()
            cfg.wavelength = LAM * 1e-10
            s = synth_sample(rng, cfg, db, phase_idx=ph_i,
                             pixel_scale=k, geom=g)
            biases.append((coarse_dist(s["img"], k) - D) * 1000)
        print(f"  k={k:5.3f} D={D*1000:4.0f}mm: "
              f"{['%+.0f' % b for b in biases]} 中位 {np.median(biases):+.1f}mm")
print("\n真实帧实测(k=2.425): 250mm→-37.1  400mm→-45.4")
