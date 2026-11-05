# -*- coding: utf-8 -*-
"""救 ceria_250 卡谷：多起点 × lr_mult 组合。"""
import sys
import os
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio
import pyFAI

from torchfai.refine import DifferentiableRefiner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ref = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3"),
                            detector="Pilatus6M", wavelength_m=9.72386e-11,
                            dist_range=[0.1, 1.0])
Hm, Wm = ref.db.extent
img = fabio.open(os.path.join(ROOT, "data/real/MX_ceria/ceria_250_1_0001.cbf")).data
p = pyFAI.load(os.path.join(ROOT, "data/real/MX_ceria/ceria_250_1_0001.poni"))
gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])

trials = [
    ("名义x800 rot/poni 3x", np.array([0.250, Hm / 2, Wm / 2, 0, 0, 0]),
     dict(iters=800, lr_mult=(1, 3, 3, 3, 3, 3))),
    ("名义 0.9x dist", np.array([0.225, Hm / 2, Wm / 2, 0, 0, 0]),
     dict(iters=800)),
    ("名义 1.1x dist", np.array([0.275, Hm / 2, Wm / 2, 0, 0, 0]),
     dict(iters=800)),
    ("真值起点 x800", gt.copy(), dict(iters=800)),
]
for tag, st, kw in trials:
    r = ref.refine(img, st, sigma_norm=None, **kw)
    px = ref.db.pixel
    print(f"{tag}: loss {r['loss_init']:.2f}->{r['loss']:.2f}  "
          f"distΔ {(r['params'][0]-gt[0])*1000:+.2f}mm  "
          f"poniΔpx {(r['params'][1]-gt[1])/px:+.1f}/{(r['params'][2]-gt[2])/px:+.1f}  "
          f"rot1Δ {(r['params'][3]-gt[3])*1000:+.2f}mrad  it={r['iters']}")
