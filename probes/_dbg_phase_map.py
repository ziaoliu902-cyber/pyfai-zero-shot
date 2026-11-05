# -*- coding: utf-8 -*-
"""调试：为什么合成图上 solve_auto 恒给 ~60mm 偏差/中心=图像中心。"""
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from torchfai.simulator import get_detector, synth_sample
from torchfai.config import SimConfig
from torchfai.phases import PHASES
from solver_core import to_canvas, solve_auto

db = get_detector("Pilatus_1M", 2)
Hm, Wm = db.extent
print("db shape", db.shape, "extent", Hm, Wm, "pixel", db.pixel)
geom = dict(dist=0.15, poni1=0.60 * Hm, poni2=0.55 * Wm, rot1=0.0, rot2=0.0, rot3=0.0)
cfg = SimConfig(detector_name="Pilatus_1M", binning=2, wavelength=1.5406e-10,
                poni_frac_range=(0.60, 0.60), rot1_range=(0, 0), rot2_range=(0, 0),
                rot3_range=(0, 0), peak_sigma_range=(0.004, 0.004), amp_jitter=(1, 1),
                dw_b_range=(0, 0), bg0_range=(0, 0), bg1_range=(0, 0), hump_prob=0,
                peak_counts_range=(10000, 10000), hot_rate_range=(0, 0),
                dead_rate_range=(0, 0), beamstop_prob=0.0, min_visible_peaks=2)
idx = list(PHASES).index("CeO2")
s = synth_sample(np.random.default_rng(1000), cfg, db, phase_idx=idx, geom=geom)
img = s["img"]
print("img", img.shape, img.dtype, "min/max/mean", img.min(), img.max(), int(img.mean()))
print("gt geom", {k: (round(v, 4) if k != "dist" else v) for k, v in s["geom"].items()})
print("gt center px", s["geom"]["poni2"] / db.pixel - 0.5, s["geom"]["poni1"] / db.pixel - 0.5)

canvas, sc, a, b = to_canvas(img)
print("to_canvas ->", canvas.shape, "s=", sc, "extra", a, b)
r = solve_auto(canvas, 1.5406, db.pixel * sc, cascade=True, auto_center=True, dbg=True)
print("solve_auto ->", None if r is None else (r[0], [round(float(x), 3) for x in r[1][:5]]))
if r is not None:
    ph, sol = r
    print("D_mm =", sol[2] * db.pixel * sc * 1000,
          "center px =", (sol[0] * sc, sol[1] * sc))
