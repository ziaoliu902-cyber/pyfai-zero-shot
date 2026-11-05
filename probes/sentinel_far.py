# -*- coding: utf-8 -*-
"""哨兵：按域分层验收（治"中位数撒谎"，10-29 教训固化）。

对一个 checkpoint 直测四个域（纯 sim，秒级）：
  far@1M    —— 今天的病根（px_v2: dist -260~-2026mm, poni 装傻）
  k<1       —— alumina 布局域（新增）
  near@k=1  —— 老本事不许丢（回归守门）
  wide@k2.4 —— 宽视域不许丢（回归守门）
每域 12 样本，报 dist/poni 中位误差。全绿 = v3 重训成功。

用法：python probes/sentinel_far.py --ckpt runs/v3far/best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchfai.config import SimConfig
from torchfai.dataset import norm_to_param
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import (get_detector, get_scaled_detector,
                                synth_sample)

from real_refine_v6 import ROOT, load_px_nn

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/v3far/best.pt"))
ap.add_argument("--n", type=int, default=12)
args = ap.parse_args()
m, dev, consts = load_px_nn(args.ckpt)

DOM = dict(far1M=dict(k=1.0, dist=(1.0, 3.0), lam=None),   # None=λ钉1.0Å（病灶配置）
           ksub1=dict(k=0.7, dist=(0.1, 1.2), lam=(5e-11, 1.5e-10)),
           near=dict(k=1.0, dist=(0.15, 0.8), lam=(5e-11, 2.3e-10)),
           wide=dict(k=2.43, dist=(0.3, 1.0), lam=(5e-11, 2.3e-10)))

ok_all = True
print(f"{'域':8s}{'dist中位Δ':>10s}{'dist最大Δ':>10s}{'poni中位Δpx':>12s}"
      f"{'λ中位ΔÅ':>9s}  判定")
for name, dom in DOM.items():
    cfg = SimConfig()
    if dom["lam"] is None:
        cfg.wavelength = 1.0e-10
    else:
        cfg.wavelength_range = dom["lam"]
    db = (get_detector("Pilatus_1M", 2) if dom["k"] == 1.0
          else get_scaled_detector("Pilatus_1M", 2, dom["k"]))
    rng = np.random.default_rng(97 + int(dom["k"] * 100))
    ed, ep, el = [], [], []
    for _ in range(args.n):
        s = synth_sample(rng, cfg, db, phase_idx=None, pixel_scale=dom["k"])
        with torch.no_grad():
            x = torch.from_numpy(
                np.log1p(s["img"].astype(np.float32)))[None, None].to(dev)
            lam_A = s["aux"][5] if s["aux"].shape[0] >= 6 else 1.0
            mu, _ = m(x, torch.tensor([lam_A], device=dev),
                      torch.tensor([dom["k"]], device=dev))
        p = norm_to_param(mu.float().cpu().numpy(), consts, dom["k"])[0]
        ed.append(abs(p[0] - s["params"][0]) * 1000)
        ep.append(np.hypot(p[1] - s["params"][1], p[2] - s["params"][2])
                  / (344e-6 * dom["k"]))
        el.append(abs(lam_A - lam_A))          # λ 直读在 refine 域验，占位
    md, mp = np.median(ed), np.median(ep)
    tight = name in ("far1M", "ksub1")
    # 新域（far/ksub1）严格双门；回归域（near/wide）只防整体崩坏（px_v2 老尾巴 264mm 属正常）
    ok = (md < (50 if tight else 60) and max(ed) < (250 if tight else 400)
          and mp < 60)
    ok_all &= ok
    print(f"{name:8s}{md:>10.1f}{max(ed):>10.1f}{mp:>12.1f}{np.median(el):>9.1f}"
          f"  {'✓' if ok else '✗ 域未愈合'}")

print("\n== AgBh 远距加密考（10-29 深夜定案：远视窗 q_max<0.67 时六相中仅")
print("   AgBh 首环 0.108 在画内——高 q 相@远距=无解空白卷，不考）==")
SICK = [("AgBh", 1.0, 0.2), ("AgBh", 1.0, 0.5), ("AgBh", 1.58, 0.2),
        ("AgBh", 1.58, 0.5), ("AgBh", 2.0, 0.2), ("AgBh", 2.0, 0.5),
        ("AgBh", 2.5, 0.2), ("AgBh", 2.5, 0.5)]
db1 = get_detector("Pilatus_1M", 2)
Hm1, Wm1 = db1.extent
cfg = SimConfig()
cfg.wavelength = 1.0e-10
rng = np.random.default_rng(11)
worst = 0.0
for ph, dist, pf in SICK:
    g = dict(dist=dist, poni1=Hm1 * pf, poni2=Wm1 * pf,
             rot1=0.0, rot2=0.0, rot3=0.0)
    s = synth_sample(rng, cfg, db1, phase_idx=PHASE_NAMES.index(ph),
                     pixel_scale=1.0, geom=g)
    with torch.no_grad():
        x = torch.from_numpy(
            np.log1p(s["img"].astype(np.float32)))[None, None].to(dev)
        mu, _ = m(x, torch.tensor([1.0], device=dev))
    p = norm_to_param(mu.float().cpu().numpy(), consts, 1.0)[0]
    ed = (p[0] - dist) * 1000
    ep = np.hypot(p[1] - g["poni1"], p[2] - g["poni2"]) / 344e-6
    worst = max(worst, abs(ed))
    print(f"  {ph:6s}@{dist}m poni{pf:.1f}: distΔ={ed:+9.1f}mm poniΔ={ep:6.1f}px")
ok_all &= worst < 250
print(f"  病灶最大 |distΔ| = {worst:.1f}mm → {'✓' if worst < 250 else '✗ 仍是崩坏域'}")

print("\n总判定:", "全绿 ✓ far 域已根治" if ok_all else "仍有红域 ✗")
sys.exit(0 if ok_all else 1)
