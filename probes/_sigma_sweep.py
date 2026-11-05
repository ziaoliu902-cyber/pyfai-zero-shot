# -*- coding: utf-8 -*-
"""σq 敏感度：GT 几何孪生 × σq 扫描 → dist 读数是否随 σq 漂移？
若漂移大且帧间真实 σq 有差 → 解释增益免疫的顽固偏差。
附带：真实帧径向剖面 vs 各 σq 孪生的剖面匹配度 → 估真实有效 σq。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI

from real_refine_v6 import load_px_nn, match_gain
from torchfai.dataset import norm_to_param
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

m, dev, consts = load_px_nn(r"runs\v4s1\best.pt")
consts = __import__("torchfai.dataset", fromlist=["TorchFAIDataset"]) \
    .TorchFAIDataset(os.path.join(ROOT, "data", "train_v4w3"),
                     "val").norm_consts
m.eval()
LAM = 0.972386
K = 2.425
PXEFF = 834.2e-6
SIGS = [0.006, 0.010, 0.015, 0.020, 0.030, 0.045, 0.060]


def coarse(canvas):
    x = torch.from_numpy(
        np.log1p(canvas[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([LAM], device=dev),
                  torch.tensor([K], device=dev))
    return float(norm_to_param(mu.double().cpu().numpy(), consts, K)[0, 0])


def radial(canvas, pr, pc):
    yy, xx = np.mgrid[0:521, 0:490]
    r = np.hypot(yy - pr, xx - pc)
    idx = np.clip((r / 2).astype(int), 0, 244)
    prof = np.bincount(idx.ravel(), weights=canvas.ravel(), minlength=245)
    cnt = np.bincount(idx.ravel(), minlength=245)
    return prof / np.maximum(cnt, 1)


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


print(f"{'帧':>4s}{'真实Δ':>8s}" + "".join(f"σ{s:.3f}" .replace("0.0", ".0")
                                        + f"{'':>4s}" for s in SIGS))
print(f"{'':>12s}" + "".join(f"{'Δmm':>7s}" for s in SIGS))
for d in (250, 300, 350, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    D = p.dist
    gt = D * 1000
    reb = np.clip(match_gain(rebin_to_canvas(img, p, K)), 0, None)
    reb = reb * auto_s(reb) * 0.5
    err_real = (coarse(reb) - D) * 1000

    db = get_scaled_detector("Pilatus_1M", 2, K)
    Hm, Wm = db.extent
    pr = p.poni1 / PXEFF
    pc = p.poni2 / PXEFF
    row = f"{d:>4d}{err_real:+8.1f}"
    g_geom = dict(dist=D, poni1=Hm * (pr / 521), poni2=Wm * (pc / 490),
                  rot1=p.rot1, rot2=p.rot2, rot3=p.rot3)
    for sg in SIGS:
        rng = np.random.default_rng(7)
        cfg = SimConfig()
        cfg.wavelength = LAM * 1e-10
        cfg.beamstop_prob = 0.0
        cfg.peak_sigma_range = (sg, sg)
        cfg.peak_sigma_broad_frac = 0.0
        s = synth_sample(rng, cfg, db,
                         phase_idx=PHASE_NAMES.index("CeO2"),
                         pixel_scale=K, geom=g_geom)
        twin = match_gain(s["img"].astype(np.float64)) \
            * auto_s(match_gain(s["img"].astype(np.float64))) * 0.5
        e = (coarse(np.clip(twin, 0, None)) - D) * 1000
        row += f"{e:+7.1f}"
    print(row)
print("""
判读：孪生Δ随σq单调漂移 → 模型σq偏置存在；对照真实Δ落在哪个σq档，
     该档即该帧有效σq；若250/300的有效σq档偏置大 → 病根=σq×dist交互。""")
