# -*- coding: utf-8 -*-
"""环宽终审：真实帧 vs 孪生的环 FWHM（细 bin 径向剖面）。"""
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

from real_refine_v6 import match_gain, to_nn_input_wide
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

PXEFF = 172e-6 / (521.0 / 2527)
LAM = 0.972386
K = 2.425


def fine_profile(canvas, pr, pc, nb=1042):
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - pr, xx - pc).ravel()
    bins = np.clip((r / r.max() * nb).astype(int), 0, nb - 1)
    prof = np.zeros(nb)
    cnt = np.zeros(nb)
    np.add.at(prof, bins, canvas.ravel())
    np.add.at(cnt, bins, 1)
    return prof / np.maximum(cnt, 1)


def ring_widths(prof, n=4):
    """峰位+FWHM（bin 单位，bin=画布半径/1042）。"""
    sm = prof - np.convolve(prof, np.ones(9) / 9, "same")
    out = []
    i = 40
    while i < len(sm) - 40 and len(out) < n:
        if sm[i] == max(sm[i - 25:i + 26]) and sm[i] > 0.08 * sm.max():
            half = sm[i] / 2
            l = i
            while l > 0 and sm[l] > half:
                l -= 1
            r = i
            while r < len(sm) - 1 and sm[r] > half:
                r += 1
            out.append((i, r - l))
            i = r + 30
        else:
            i += 1
    return out


for d in (250, 400):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    wide, _, extra = to_nn_input_wide(img)
    wide = match_gain(wide).astype(np.float64)
    pr = p.poni1 / PXEFF
    pc = (p.poni2 - extra) / PXEFF
    prof_r = fine_profile(wide, pr, pc)

    db = get_scaled_detector("Pilatus_1M", 2, K)
    rng = np.random.default_rng(7)
    cfg = SimConfig()
    cfg.wavelength = LAM * 1e-10
    s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                     pixel_scale=K,
                     geom=dict(dist=p.dist, poni1=p.poni1, poni2=p.poni2,
                               rot1=0.0, rot2=0.0, rot3=0.0))
    prof_t = fine_profile(s["img"].astype(np.float64), pr, pc)

    wr = ring_widths(prof_r)
    wt = ring_widths(prof_t)
    bin_mm = (np.hypot(pr, pc) / 1042) * PXEFF * 1000
    print(f"\n== ceria{d} 环宽(FWHM, px) ==")
    print(f"  真实: {[(int(i*521/1042*0), w) for i, w in wr]}")
    print(f"  孪生: {wt}")
    if wr and wt:
        print(f"  真实前环宽 {wr[0][1]} bins vs 孪生 {wt[0][1]} bins "
              f"（比 {wr[0][1]/max(wt[0][1],1):.1f}×）")
