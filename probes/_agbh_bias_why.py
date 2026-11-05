# -*- coding: utf-8 -*-
"""AgBh 远距 +460 系统偏差根因分离（单变量对照）。

嫌疑：a) 束流挡板吞前几环（远距首环 r=17~43px，挡板可达 45px）
      b) 宽峰环心外偏
      c) λ 条件化对低 q 相失灵
      d) 背景干扰
每配置 4 样本 @1.58m/poni0.5/rot0，看 dist 偏差随哪个开关消失。
"""
import sys
import os

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_detector, synth_sample
from torchfai.dataset import norm_to_param
from real_refine_v6 import load_px_nn

m, dev, consts = load_px_nn(r"runs/v3far2/best.pt")
db = get_detector("Pilatus_1M", 2)
Hm, Wm = db.extent
ph = PHASE_NAMES.index("AgBh")


def bias(ov, lam=1.0, n=4, seed=11, dist=1.58):
    cfg = SimConfig()
    cfg.wavelength = lam * 1e-10
    for k, v in ov.items():
        setattr(cfg, k, v)
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        g = dict(dist=dist, poni1=Hm * 0.5, poni2=Wm * 0.5,
                 rot1=0.0, rot2=0.0, rot3=0.0)
        s = synth_sample(rng, cfg, db, phase_idx=ph, pixel_scale=1.0, geom=g)
        with torch.no_grad():
            x = torch.from_numpy(
                np.log1p(s["img"].astype(np.float32)))[None, None].to(dev)
            mu, _ = m(x, torch.tensor([lam], device=dev))
        p = norm_to_param(mu.float().cpu().numpy(), consts, 1.0)[0]
        errs.append((p[0] - dist) * 1000)
    return errs


EXPS = [
    ("A 基线", {}),
    ("B 小挡板8-10px", dict(beamstop_r_range=(8.0, 10.0))),
    ("C 窄峰only", dict(peak_sigma_broad_frac=0.0)),
    ("D 窄峰+小挡板", dict(peak_sigma_broad_frac=0.0,
                          beamstop_r_range=(8.0, 10.0))),
    ("F 无挡板极限", dict(beamstop_r_range=(0.01, 0.02))),
    ("G 无热点", dict(hot_rate_range=(0.0, 0.0))),
]
for name, ov in EXPS:
    e = bias(ov)
    print(f"{name:14s} dist偏差(mm): " + " ".join(f"{v:+7.0f}" for v in e))

print()
for lam in (0.6, 0.8, 1.2):
    e = bias({}, lam=lam)
    print(f"λ={lam}Å       dist偏差(mm): " + " ".join(f"{v:+7.0f}" for v in e))
