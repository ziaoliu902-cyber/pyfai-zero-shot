# -*- coding: utf-8 -*-
"""疑点复核：AgBh 远距偏差到底是 λ 稀薄还是 rot=0 流形？

今晚所有"有病"测量（病灶八联/λ扫描）全用 rot=0；
所有"健康"测量（far1M 随机域 14mm）全是随机转角。
对照：AgBh@1.58m/λ1.0/poni0.5，各 8 个种子：
  组1 rot=0           组2 |rot|≤0.01        组3 |rot|≤0.05
看偏差是否随"离开零转角流形"消失。
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


def run(rot_scale, n=8):
    cfg = SimConfig()
    cfg.wavelength = 1.0e-10
    errs = []
    for seed in range(100, 100 + n):
        rng = np.random.default_rng(seed)
        r = rot_scale
        g = dict(dist=1.58,
                 poni1=Hm * 0.5, poni2=Wm * 0.5,
                 rot1=rng.uniform(-r, r), rot2=rng.uniform(-r, r),
                 rot3=rng.uniform(-r, r))
        s = synth_sample(rng, cfg, db, phase_idx=ph, pixel_scale=1.0, geom=g)
        with torch.no_grad():
            x = torch.from_numpy(
                np.log1p(s["img"].astype(np.float32)))[None, None].to(dev)
            mu, _ = m(x, torch.tensor([1.0], device=dev))
        p = norm_to_param(mu.float().cpu().numpy(), consts, 1.0)[0]
        errs.append((p[0] - 1.58) * 1000)
    return errs


for name, r in (("rot=0（完美圆）", 0.0), ("|rot|≤0.01", 0.01),
                ("|rot|≤0.05", 0.05), ("|rot|≤0.07(全域)", 0.07)):
    e = run(r)
    med = np.median(e)
    print(f"{name:16s} 中位 {med:+7.0f}mm  范围 [{min(e):+6.0f}, {max(e):+6.0f}]"
          f"  明细: " + " ".join(f"{v:+.0f}" for v in e))
