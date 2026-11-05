# -*- coding: utf-8 -*-
"""环宽因果确认：孪生 σq=0.002~0.012(训练窄档) vs 0.015~0.03(真实档)
vs 0.05~0.6(训练宽档)。真实档复现 -40mm → 环宽缝隙=病根。"""
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
K = 2.425


def coarse(canvas):
    x = torch.from_numpy(
        np.log1p(canvas[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([LAM], device=dev),
                  torch.tensor([K], device=dev))
    return float(norm_to_param(mu.double().cpu().numpy(), consts, K)[0, 0])


db = get_scaled_detector("Pilatus_1M", 2, K)
ph_i = PHASE_NAMES.index("CeO2")
print("σq 档位扫描（GT=D, 钉几何, k=2.425）:")
for tag, sr in (("窄(训练域)", (0.002, 0.012)),
                ("真实档", (0.015, 0.03)),
                ("宽(训练域)", (0.05, 0.15))):
    for D in (0.2492, 0.3995):
        rng = np.random.default_rng(7)
        cfg = SimConfig()
        cfg.wavelength = LAM * 1e-10
        cfg.peak_sigma_range = sr
        cfg.peak_sigma_broad_frac = 0.0
        biases = []
        for t in range(3):
            s = synth_sample(rng, cfg, db, phase_idx=ph_i,
                             pixel_scale=K,
                             geom=dict(dist=D, poni1=db.extent[0] * 0.5,
                                       poni2=db.extent[1] * 0.5,
                                       rot1=0.0, rot2=0.0, rot3=0.0))
            biases.append((coarse(s["img"]) - D) * 1000)
        print(f"  {tag:10s} σq{sr} D={D*1000:3.0f}mm: "
              f"{['%+.0f' % b for b in biases]} 中位 {np.median(biases):+.1f}mm")
print("\n真实帧: 250mm→-37.1  400mm→-45.4")
