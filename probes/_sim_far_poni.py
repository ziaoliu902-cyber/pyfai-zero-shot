# -*- coding: utf-8 -*-
"""sim 远距 poni 能力测试：模型在训练分布内的远距 poni 到底准不准。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
from real_refine_v6 import ROOT, load_px_nn
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

m, dev, consts = load_px_nn(os.path.join(ROOT, "runs/px_v2/best.pt"))
cfg = SimConfig()
cfg.wavelength = 1.0e-10
db = get_scaled_detector("Pilatus_1M", 2, 1.0)
Hm, Wm = db.extent
rng = np.random.default_rng(11)

print(f"{'相':8s}{'dist':>6s}{'poni真值(px)':>16s}{'poniNN(px)':>14s}"
      f"{'Δpx':>12s}{'distΔmm':>9s}")
for phase in ("AgBh", "CeO2"):
    for dist in (1.58, 2.5):
        for pon_frac in (0.2, 0.5):
            pi = PHASE_NAMES.index(phase)
            g = dict(dist=dist, poni1=Hm * pon_frac, poni2=Wm * pon_frac,
                     rot1=0.0, rot2=0.0, rot3=0.0)
            s = synth_sample(rng, cfg, db, phase_idx=pi, pixel_scale=1.0,
                             geom=g)
            with torch.no_grad():
                x = torch.from_numpy(
                    np.log1p(s["img"].astype(np.float32)))[None, None].to(dev)
                mu, _ = m(x, torch.tensor([1.0], device=dev),
                          torch.tensor([1.0], device=dev))
            from torchfai.dataset import norm_to_param
            p = norm_to_param(mu.float().cpu().numpy(), consts, 1.0)[0]
            py, pxn = p[1] / 344e-6, p[2] / 344e-6
            gy, gx = g["poni1"] / 344e-6, g["poni2"] / 344e-6
            print(f"{phase:8s}{dist:>6.2f}{gy:>8.0f},{gx:>7.0f}"
                  f"{py:>7.0f},{pxn:>6.0f}"
                  f"{np.hypot(py-gy, pxn-gx):>12.0f}"
                  f"{(p[0]-dist)*1000:>+9.1f}")
