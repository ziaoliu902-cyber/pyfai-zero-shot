# -*- coding: utf-8 -*-
"""目视终审：ceria_350 真实画布 vs GT 孪生 并排 + 残差 + 方位角结构。
不再猜假设——把皮撕开看。"""
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_refine_v6 import match_gain, to_nn_input_wide
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

LAM = 0.972386
K = 2.425
PXEFF = 172e-6 / (521.0 / 2527)

img = fabio.open(os.path.join(
    ROOT, "data", "real", "MX_ceria/ceria_350_1_0001.cbf")).data
p = pyFAI.load(os.path.join(
    ROOT, "data", "real", "MX_ceria/ceria_350_1_0001.poni"))
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

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for ax, (im_, ttl) in zip(axes.ravel(), [
        (np.log1p(wide), f"真实 ceria_350  log1p  GT={D*1000:.1f}mm"),
        (np.log1p(twin), "GT 孪生 log1p（同几何/λ/相/增益）"),
        (np.log1p(wide) - np.log1p(twin),
         "残差 log(real)−log(twin)  ← 皮在哪"),
        (None, "方位角切片对比 r=180±3px")]):
    if im_ is not None:
        v = np.percentile(im_, [1, 99])
        ax.imshow(im_, cmap="gray", vmin=v[0], vmax=v[1])
        ax.set_title(ttl)
        ax.plot([pc], [pr], "r+", ms=12)
    else:
        yy, xx = np.mgrid[0:521, 0:490]
        rr = np.hypot(yy - pr, xx - pc)
        band = (rr > 177) & (rr < 183)
        vals_r = np.log1p(wide[band])
        vals_t = np.log1p(twin[band])
        seg = 36
        idx = (np.arctan2(yy[band] - pr, xx[band] - pc) + np.pi) / (
            2 * np.pi) * seg
        ar = np.array([vals_r[idx == i].mean() for i in range(seg)])
        at = np.array([vals_t[idx == i].mean() for i in range(seg)])
        ax.plot(ar, "r-", lw=2, label="真实")
        ax.plot(at, "b-", lw=2, label="孪生")
        ax.set_title(ttl)
        ax.legend()
        ax.set_xlabel("方位角槽 36 段")
        ax.set_ylabel("log1p 亮度")
fig.suptitle("ceria_350 真实 vs GT孪生 —— 皮在哪", fontsize=13)
fig.tight_layout()
out = os.path.join(ROOT, "probes", "results", "_visual_diff.png")
fig.savefig(out, dpi=110)
print("saved:", out)
