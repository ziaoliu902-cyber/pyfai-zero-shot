# -*- coding: utf-8 -*-
"""信息含量审计：卡住样本 vs 正常样本，在真值几何下各能看到多少条环。

若两组可见环数相当 → 图里有信息，卡住=学习问题（补课可修）
若卡住组环显著更少 → 信息不足，属于图像极限（σ 报警是正解）
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset
from torchfai.phases import PHASE_NAMES, phase_qi
from torchfai.simulator import ring_radius_m

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "data/train_v2/meta.json"),
                     encoding="utf-8"))["config"]
PIX = cfg.get("pixel_size_um", 172.0) * cfg["binning"] * 1e-6
Hm, Wm = 521 * PIX, 490 * PIX
lam_A = cfg["wavelength"] * 1e10
BS = (max(cfg["beamstop_r_range"]) + 3.0) * PIX if cfg.get("beamstop_r_range") else 25 * PIX

def n_rings(ph, p):
    """真值几何下可见环数（保守：最大挡光盘）。"""
    qs, _ = phase_qi(ph)
    r_far = 1.05 * float(np.hypot(0.75 * Hm, 0.75 * Wm))
    rr = ring_radius_m(qs, p[0], lam_A)
    vis = (rr > BS) & (rr < r_far)
    return int(vis.sum()), float(rr[vis].min()) / PIX if vis.any() else 0.0

ds = TorchFAIDataset(os.path.join(ROOT, "data/train_v2"), split="test")
rows = [l.strip().split(",") for l in open(os.path.join(ROOT, "runs/refine_v2/results.csv"),
       encoding="utf-8").readlines()[1:] if l.strip()]
stuck_ks = {int(r[0]) for r in rows if float(r[9]) > 5.0}
ok_ks = {int(r[0]) for r in rows if float(r[9]) < 0.5}

rng = np.random.default_rng(7)
def sample_group(ks, n=120):
    out = []
    for k in sorted(ks):
        g = int(ds._sel[k])
        ph = PHASE_NAMES[int(ds._maux[g][0])]
        out.append((k, ph, ds._mparams[g].astype(np.float64)))
        if len(out) >= n:
            break
    return out

for tag, grp in (("卡住(>5mm)", sample_group(stuck_ks)),
                 ("精准(<0.5mm)", sample_group(ok_ks))):
    ns, rmin = [], []
    for k, ph, p in grp:
        n, r0 = n_rings(ph, p)
        ns.append(n); rmin.append(r0)
    ns, rmin = np.array(ns), np.array(rmin)
    print(f"{tag}: N={len(grp)}")
    print(f"  可见环数: 中位 {np.median(ns):.0f}  分布 "
          f"({np.percentile(ns,10):.0f}/{np.percentile(ns,50):.0f}/{np.percentile(ns,90):.0f})")
    print(f"  最内环半径: 中位 {np.median(rmin):.0f} px")
    print(f"  环数≤3 的占比: {(ns<=3).mean()*100:.0f}%")
