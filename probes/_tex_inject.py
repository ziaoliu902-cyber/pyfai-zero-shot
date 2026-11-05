# -*- coding: utf-8 -*-
"""方位角纹理因果实验：孪生环 × 真实帧方位角调制 → dist 读数掉不掉？
机制假说：真实环是残缺弧（择优取向/样品纹理），仿真全环。
若注入后孪生复现真实Δ → 纹理=病因 → 仿真加随机方位调制增广即可治。
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

m, dev, consts = load_px_nn(r"runs\v4w2\best.pt")
consts = __import__("torchfai.dataset", fromlist=["TorchFAIDataset"]) \
    .TorchFAIDataset(os.path.join(ROOT, "data", "train_v4w2"),
                     "val").norm_consts
m.eval()
LAM = 0.972386
K = 2.425
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


def coarse(canvas):
    x = torch.from_numpy(
        np.log1p(np.clip(canvas, 0, None))[None, None].astype(np.float32)
    ).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([LAM], device=dev),
                  torch.tensor([K], device=dev))
    return float(norm_to_param(mu.double().cpu().numpy(), consts, K)[0, 0])


yy, xx = np.mgrid[0:521, 0:490]

print(f"{'帧':>4s}{'真实Δ':>8s}{'孪生Δ':>8s}{'纹理孪生Δ':>10s}")
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
    twin = match_gain(s["img"].astype(np.float64))
    twin = twin * auto_s(twin) * 0.5
    err_twin = (coarse(twin) - D) * 1000

    # ---- 纹理注入：逐环带乘真实方位剖面/该带均值 ----
    rr = np.hypot(yy - pr, xx - pc)
    ang = ((np.arctan2(yy - pr, xx - pc) + np.pi) / (2 * np.pi) * 72
           ).astype(int)
    tex = twin.copy()
    NB = 72
    # 真实帧逐 2px 径向带的方位剖面
    idx = np.clip((rr / 2).astype(int), 0, 244)

    def az_profile(canvas, band):
        v = canvas[band]
        a = ang[band]
        out = np.zeros(NB)
        cnt = np.zeros(NB)
        np.add.at(out, a, v)
        np.add.at(cnt, a, 1)
        return out / np.maximum(cnt, 1)

    bands = [(lo, lo + 2) for lo in range(10, 240, 2)]
    for lo, hi_ in bands:
        band = (rr >= lo) & (rr < hi_)
        if band.sum() < 200:
            continue
        az_r = az_profile(reb, band)
        mu_r = az_r.mean()
        if mu_r < 1e-6:
            continue
        fac = az_r / mu_r                    # 真实纹理因子（均值1）
        tex[band] *= fac[ang[band]]
    err_tex = (coarse(tex) - D) * 1000

    print(f"{d:>4d}{err_real:+8.1f}{err_twin:+8.1f}{err_tex:+10.1f}")
print("""
判读：纹理孪生Δ → 真实Δ 一带（尤其 250）→ 方位纹理=病因，仿真增广治本。""")
