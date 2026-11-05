# -*- coding: utf-8 -*-
"""亮度配平后的 250 vs 350 径向解剖：环峰高度/背景形状/对比度逐项对比。
auto-gain 后中位对齐 → 差异不再被亮度淹没。
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

from real_refine_v6 import match_gain
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

K = 2.425
LAM = 0.972386
PXEFF = 834.2e-6

src = open(os.path.join(ROOT, "probes", "_rebin_test.py"),
           encoding="utf-8").read()
g = {}
exec(compile(src.split("print(f\"")[0], "rebin_fn", "exec"), g)
rebin_to_canvas = g["rebin_to_canvas"]
MED_LOG_REF = 6.68


def auto_s(canvas):
    v = canvas[canvas > 0]
    lo, hi = 0.05, 64.0
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        if float(np.median(np.log1p(v * mid))) - MED_LOG_REF < 0:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


yy, xx = np.mgrid[0:521, 0:490]


def radial_log(canvas, pr, pc):
    r = np.hypot(yy - pr, xx - pc)
    idx = np.clip((r / 4).astype(int), 0, 122)   # 4px/桶
    L = np.log1p(canvas)
    prof = np.bincount(idx.ravel(), weights=L.ravel(), minlength=123)
    cnt = np.bincount(idx.ravel(), minlength=123)
    return prof / np.maximum(cnt, 1)


for d in (250, 350):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    D = p.dist
    reb = np.clip(match_gain(rebin_to_canvas(img, p, K)), 0, None)
    reb = reb * auto_s(reb) * 0.5
    pr = p.poni1 / PXEFF
    pc = p.poni2 / PXEFF

    db = get_scaled_detector("Pilatus_1M", 2, K)
    Hm, Wm = db.extent
    rng = np.random.default_rng(7)
    cfg = SimConfig()
    cfg.wavelength = LAM * 1e-10
    cfg.beamstop_prob = 0.0
    cfg.peak_sigma_range = (0.02, 0.02)
    cfg.peak_sigma_broad_frac = 0.0
    gg = dict(dist=D, poni1=Hm * (pr / 521), poni2=Wm * (pc / 490),
              rot1=p.rot1, rot2=p.rot2, rot3=p.rot3)
    s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                     pixel_scale=K, geom=gg)
    tw = match_gain(s["img"].astype(np.float64))
    tw = tw * auto_s(tw) * 0.5

    fr = radial_log(reb, pr, pc)
    ft = radial_log(tw, pr, pc)
    print(f"\n===== ceria_{d}（4px/桶，log1p 值）=====")
    print(f"{'半径px':>8s}{'真实':>8s}{'孪生':>8s}{'差':>8s}")
    for i in range(6, 123, 4):
        bar = "#" * int(max(fr[i] - min(fr[i], ft[i]), 0) * 6)
        print(f"{i*4:>8d}{fr[i]:>8.2f}{ft[i]:>8.2f}{fr[i]-ft[i]:>+8.2f} {bar}")
    # 峰谷统计
    pk = [i for i in range(6, 117)
          if ft[i] == max(ft[i - 2:i + 3])]
    if pk:
        contrast_r = np.mean([fr[i] - np.median(fr) for i in pk])
        contrast_t = np.mean([ft[i] - np.median(ft) for i in pk])
        print(f"环峰对比(峰-中位): 真实 {contrast_r:.2f} vs 孪生 {contrast_t:.2f}"
              f"   中位: {np.median(fr):.2f}/{np.median(ft):.2f}")
