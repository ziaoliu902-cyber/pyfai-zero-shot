# -*- coding: utf-8 -*-
"""中心偏移精测 v2：扩范围 ±14px 步长 0.75 + 估计器自校验。
自校验：孪生 vs 注入已知平移的孪生 → 复原值须≈注入值，否则估计器不可信。
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
GRID = np.arange(-14, 14.01, 0.75)


def best_shift(A, B):
    best = (1e18, 0.0, 0.0)
    for dr in GRID:
        for dc in GRID:
            Bs = ndshift(B, (dr, dc), order=1, mode="nearest")
            r = float(np.mean((A - Bs) ** 2))
            if r < best[0]:
                best = (r, dr, dc)
    return best[1], best[2]


img0 = fabio.open(os.path.join(
    ROOT, "data", "real", "MX_ceria/ceria_350_1_0001.cbf")).data
p0 = pyFAI.load(os.path.join(
    ROOT, "data", "real", "MX_ceria/ceria_350_1_0001.poni"))
D0 = p0.dist
wide0, k0, extra0 = to_nn_input_wide(img0)
pr = p0.poni1 / PXEFF
pc = (p0.poni2 - extra0) / PXEFF
db = get_scaled_detector("Pilatus_1M", 2, K)
Hm, Wm = db.extent
rng = np.random.default_rng(7)
g = dict(dist=D0, poni1=Hm * (pr / 521), poni2=Wm * (pc / 490),
         rot1=0.0, rot2=0.0, rot3=0.0)
cfg = SimConfig()
cfg.wavelength = LAM * 1e-10
cfg.beamstop_prob = 0.0
s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                 pixel_scale=K, geom=g)
twin0 = np.log1p(match_gain(s["img"].astype(np.float64)))
real0 = np.log1p(wide0)

print("== 自校验：注入 (+4.5, -3.0) 到孪生，看能否复原 ==")
inj = ndshift(np.exp(twin0) - 1, (4.5, -3.0), order=1, mode="nearest")
inj = np.log1p(np.clip(inj, 0, None))
print("复原:", best_shift(twin0, inj))

print("\n== 真实 vs 孪生（三帧，精测） ==")
for d in (250, 350, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    wide, kk, extra = to_nn_input_wide(img)
    A = np.log1p(wide)
    dr, dc = best_shift(A, twin0)
    print(f"ceria{d}: 最优平移 (Δrow,Δcol)=({dr:+.2f},{dc:+.2f})px"
          f"  |offset|={np.hypot(dr, dc):.1f}px"
          f"  = {np.hypot(dr, dc)*PXEFF*1000:.1f}mm 物理量")
