# -*- coding: utf-8 -*-
"""_sigq_calib.py：σq 标签标定——已知 σq 的孪生量 w(FWHM/r)，验单调性。
w 若不随 σq 单调 → 标签被背景/驼峰污染，须修 rel_width 再重打标签。
附带：真实 ceria 帧的 w 落在 sim 域哪一档。
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

from real_refine_v6 import to_nn_input_wide
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

sys.path.insert(0, os.path.join(ROOT, "probes"))
from gen_sigq_labels import rel_width

LAM = 0.972386
K = 2.425
PXEFF = 172e-6 / (521.0 / 2527)
SIGS = [0.006, 0.010, 0.015, 0.020, 0.030, 0.045, 0.060]

print("sim 孪生标定（ceria250 几何，CeO2）：")
print(f"{'σq':>7}{'w=FWHM/r':>10}")
for d in (250, 450):
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    db = get_scaled_detector("Pilatus_1M", 2, K)
    Hm, Wm = db.extent
    pr = p.poni1 / PXEFF
    pc = p.poni2 / PXEFF
    g_geom = dict(dist=p.dist, poni1=Hm * (pr / 521), poni2=Wm * (pc / 490),
                  rot1=p.rot1, rot2=p.rot2, rot3=p.rot3)
    print(f"-- d={d} --")
    for sg in SIGS:
        rng = np.random.default_rng(7)
        cfg = SimConfig()
        cfg.wavelength = LAM * 1e-10
        cfg.beamstop_prob = 0.0
        cfg.peak_sigma_range = (sg, sg)
        cfg.peak_sigma_broad_frac = 0.0
        s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                         pixel_scale=K, geom=g_geom)
        img = s["img"].astype(np.float64)
        w = rel_width(img, pr, pc)
        print(f"{sg:7.3f}{w:10.4f}")

print("\n真实帧 w（to_nn_input_wide 路径）：")
for d in (150, 250, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    wide, k, extra = to_nn_input_wide(img)
    pr = p.poni1 / PXEFF
    pc = (p.poni2 - extra) / PXEFF
    w = rel_width(wide.astype(np.float64), pr, pc)
    print(f"ceria{d}: w={w:.4f}")
