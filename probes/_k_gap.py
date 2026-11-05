# -*- coding: utf-8 -*-
"""k 档位缝假说：孪生用受过训的档(k=2.5/2.0) vs 缝里(k=2.425)。
档上准、缝里偏 → px 触须只在离散 k 上训过，缝=外推坑。解=连续 k 重训。"""
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
from torchfai.simulator import get_detector, synth_sample

m, dev, consts = load_px_nn(r"runs\v3far4\last.pt")
m.eval()


def coarse_dist(canvas, k, lam_A):
    x = torch.from_numpy(
        np.log1p(canvas[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([lam_A], device=dev),
                  torch.tensor([k], device=dev))
    return float(norm_to_param(mu.double().cpu().numpy(), consts, k)[0, 0])


db = get_detector("Pilatus_1M", 2)
ph_i = PHASE_NAMES.index("CeO2")
D = 0.2492
LAM = 0.972386

for k in (2.0, 2.425, 2.5):
    cfg = SimConfig()
    cfg.dist_range = (D * 0.999, D * 1.001)
    cfg.wavelength_range = (LAM * 1e-10, LAM * 1e-10)
    cfg.poni_frac_range = (0.45, 0.55)
    rng = np.random.default_rng(7)
    biases = []
    for t in range(3):
        try:
            s = synth_sample(rng, cfg, db, phase_idx=ph_i, pixel_scale=k)
        except RuntimeError:
            print(f"k={k}: 该几何下无可行组合（环出界）")
            break
        biases.append((coarse_dist(s["img"], k, LAM) - D) * 1000)
    if biases:
        print(f"k={k:5.3f}: 孪生偏差 {['%+.0f' % b for b in biases]} "
              f"中位 {np.median(biases):+.1f}mm")
print("\n真实帧 k=2.425 实测偏: -37(250mm) ~ -45(400mm)")
print("若 k=2.5 准 而 2.425 偏 → 档位缝=病根 → 解:连续 k 增补重训")
