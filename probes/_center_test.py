# -*- coding: utf-8 -*-
"""中心映射终审：真实画布 vs GT孪生 亚像素互相关 → 最优平移量。
若最优平移 ≈ (0,0) → 中心映射无罪；若稳定偏离几px → pr/pc 映射有系统差。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI
from scipy.ndimage import shift as ndshift

from real_refine_v6 import match_gain, to_nn_input_wide
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

LAM = 0.972386
K = 2.425
PXEFF = 172e-6 / (521.0 / 2527)

for d in (250, 350, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    D = p.dist
    wide, k, extra = to_nn_input_wide(img)
    pr = p.poni1 / PXEFF
    pc = (p.poni2 - extra) / PXEFF

    db = get_scaled_detector("Pilatus_1M", 2, K)
    Hm, Wm = db.extent
    rng = np.random.default_rng(7)
    g = dict(dist=D, poni1=Hm * (pr / 521), poni2=Wm * (pc / 490),
             rot1=0.0, rot2=0.0, rot3=0.0)
    cfg = SimConfig()
    cfg.wavelength = LAM * 1e-10
    cfg.beamstop_prob = 0.0
    s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                     pixel_scale=K, geom=g)
    twin = match_gain(s["img"].astype(np.float64))

    A = np.log1p(wide)
    best = (1e18, 0.0, 0.0)
    for dr in np.arange(-6, 6.01, 1.5):
        for dc in np.arange(-6, 6.01, 1.5):
            Tw = np.log1p(ndshift(twin, (dr, dc), order=1, mode="nearest"))
            r = float(np.mean((A - Tw) ** 2))
            if r < best[0]:
                best = (r, dr, dc)
    print(f"ceria{d}: 最优孪生平移 (Δrow,Δcol)=({best[1]:+.1f},{best[2]:+.1f})px"
          f"  mse={best[0]:.4f}  （0,0 附近=映射无罪）")
