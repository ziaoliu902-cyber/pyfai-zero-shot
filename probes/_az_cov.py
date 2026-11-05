# -*- coding: utf-8 -*-
"""方位角覆盖率解剖：真实环是全环还是残缺弧？
对每帧 rebin 画布，各环带算 36 槽方位剖面 → 覆盖率(>0.3×环峰的槽占比) + 槽间CV。
孪生(完美全环)做对照。残缺弧=模型读近的候选机制。
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


def ring_az(canvas, pr, pc):
    """返回 [(r_px, 覆盖率, CV), ...] 每个可检环。"""
    r = np.hypot(yy - pr, xx - pc)
    idx = np.clip((r / 2).astype(int), 0, 244)
    prof = np.bincount(idx.ravel(), weights=canvas.ravel(), minlength=245)
    cnt = np.bincount(idx.ravel(), minlength=245)
    prof = prof / np.maximum(cnt, 1)
    bg = np.median(prof)
    out = []
    ang = ((np.arctan2(yy - pr, xx - pc) + np.pi) / (2 * np.pi) * 36
           ).astype(int)
    for i in range(10, 235):
        if prof[i] == max(prof[i - 3:i + 4]) and prof[i] > 3 * bg:
            if out and i - out[-1][0] <= 4:
                continue
            band = (r > (i - 1) * 2) & (r < (i + 1) * 2)
            v = canvas[band]
            a = ang[band]
            az = np.array([v[a == j].mean() for j in range(36)])
            az = az[np.isfinite(az)]
            pk = az.max()
            cov = float((az > 0.3 * pk).mean())
            cv = float(az.std() / (az.mean() + 1e-9))
            out.append((i * 2, cov, cv))
    return out


print(f"{'帧':>4s}{'n环':>5s}{'均覆盖':>8s}{'最差覆盖':>9s}{'均CV':>7s}"
      f"   孪生对照: n环 均覆盖 均CV")
db = get_scaled_detector("Pilatus_1M", 2, K)
Hm, Wm = db.extent
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    D = p.dist
    reb = np.clip(match_gain(rebin_to_canvas(img, p, K)), 0, None)
    reb = reb * auto_s(reb) * 0.5
    pr = p.poni1 / PXEFF
    pc = p.poni2 / PXEFF
    rings = ring_az(reb, pr, pc)
    if not rings:
        print(f"{d:>4d}  无可检环")
        continue
    covs = [x[1] for x in rings]
    cvs = [x[2] for x in rings]

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
    trings = ring_az(tw, pr, pc)
    tcovs = [x[1] for x in trings] or [1]
    tcvs = [x[2] for x in trings] or [0]
    print(f"{d:>4d}{len(rings):>5d}{np.mean(covs):>8.2f}"
          f"{min(covs):>9.2f}{np.mean(cvs):>7.2f}"
          f"   {'':>3s}{len(trings):>3d} {np.mean(tcovs):>5.2f}"
          f" {np.mean(tcvs):>5.2f}")
