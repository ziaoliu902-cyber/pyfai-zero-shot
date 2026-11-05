# -*- coding: utf-8 -*-
"""决定性实验：AgBh 远距 D×λ 简并的分账方式。

固定图像（D=1.58m, λ=1.0Å），扫 λ 条件输入 0.6~1.5：
  - 若 D_pred×λ_in ≈ 常数 → 模型把"半径标度"读对了，分账全靠 λ 输入
  - 若 D_pred 对 λ_in 不动 → λ 条件化被无视（触须失灵）
  - 中间态 → 部分失灵（α 指数）
再对照：高 q 相（LaB6）同实验作基准。
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

for ph_name, D, lam_true in (("AgBh", 1.58, 1.0), ("LaB6", 0.30, 1.0)):
    ph = PHASE_NAMES.index(ph_name)
    cfg = SimConfig()
    cfg.wavelength = lam_true * 1e-10
    rng = np.random.default_rng(11)
    g = dict(dist=D, poni1=Hm * 0.5, poni2=Wm * 0.5,
             rot1=0.0, rot2=0.0, rot3=0.0)
    s = synth_sample(rng, cfg, db, phase_idx=ph, pixel_scale=1.0, geom=g)
    x = torch.from_numpy(np.log1p(s["img"].astype(np.float32)))[None, None].to(dev)
    print(f"\n== {ph_name} 图像固定 D={D}m λ={lam_true}Å，扫 λ 条件输入 ==")
    print(f"{'λ_in':>5s} {'D_pred(m)':>10s} {'D_pred×λ_in':>12s} {'偏差mm':>8s}")
    for lam_in in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5):
        with torch.no_grad():
            mu, _ = m(x, torch.tensor([lam_in], device=dev))
        p = norm_to_param(mu.float().cpu().numpy(), consts, 1.0)[0]
        print(f"{lam_in:5.2f} {p[0]:10.3f} {p[0]*lam_in:12.3f} "
              f"{(p[0]-D)*1000:+8.0f}")
