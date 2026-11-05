# -*- coding: utf-8 -*-
"""合成图径向谱 vs 预期环半径：判断是"环不在预期位置"还是"找峰/相判失败"。"""
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from torchfai.simulator import get_detector, synth_sample
from torchfai.config import SimConfig
from torchfai.phases import PHASES, phase_qs
from solver_core import to_canvas

db = get_detector("Pilatus_1M", 2)
Hm, Wm = db.extent
geom = dict(dist=0.15, poni1=0.60 * Hm, poni2=0.55 * Wm, rot1=0.0, rot2=0.0, rot3=0.0)
cfg = SimConfig(detector_name="Pilatus_1M", binning=2, wavelength=1.5406e-10,
                poni_frac_range=(0.60, 0.60), rot1_range=(0, 0), rot2_range=(0, 0),
                rot3_range=(0, 0), peak_sigma_range=(0.004, 0.004), amp_jitter=(1, 1),
                dw_b_range=(0, 0), bg0_range=(0, 0), bg1_range=(0, 0), hump_prob=0,
                peak_counts_range=(10000, 10000), hot_rate_range=(0, 0),
                dead_rate_range=(0, 0), beamstop_prob=0.0)
idx = list(PHASES).index("CeO2")
s = synth_sample(np.random.default_rng(1000), cfg, db, phase_idx=idx, geom=geom)
img = s["img"].astype(float)
canvas, sc, _, _ = to_canvas(img)
print("canvas", canvas.shape, "s", sc)
gtx, gty = s["geom"]["poni2"] / db.pixel - 0.5, s["geom"]["poni1"] / db.pixel - 0.5
print("gt center native", gtx, gty, "canvas", gtx / sc, gty / sc)

H, W = canvas.shape
yy, xx = np.mgrid[0:H, 0:W]
r = np.hypot(yy - gty / sc, xx - gtx / sc)
dr = 0.5
bins = np.arange(0, min(H, W) / 1.5, dr)
p = np.zeros(len(bins) - 1)
c = np.zeros(len(bins) - 1)
idxb = np.digitize(r.ravel(), bins) - 1
ok = (idxb >= 0) & (idxb < len(bins) - 1)
np.add.at(p, idxb[ok], canvas.ravel()[ok])
np.add.at(c, idxb[ok], 1)
p = p / np.maximum(c, 1)
from scipy.signal import find_peaks
pk, props = find_peaks(p, height=p.max() * 0.05, distance=3)
print("检测峰半径(canvas px):", [round(float(bins[i]), 1) for i in pk],
      "峰高:", [int(x) for x in props["peak_heights"]])

lam = 1.5406
qs = phase_qs("CeO2")
tt = 2 * np.arcsin(np.clip(lam * np.asarray(qs) / (4 * np.pi), 0, 0.999))
radii = 0.15 * np.tan(tt) / db.pixel / sc
print("预期 CeO2 环半径(canvas px):", np.round(radii, 1))
tt2 = 2 * np.arcsin(np.clip(lam * np.asarray(phase_qs("Al2O3")) / (4 * np.pi), 0, 0.999))
print("Al2O3@130mm 预期:", np.round(0.130 * np.tan(tt2) / db.pixel / sc, 1))
