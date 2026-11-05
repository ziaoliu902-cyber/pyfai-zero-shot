# -*- coding: utf-8 -*-
"""按可见环数分层的卡住率：区分"实战工况"与"全随机压力测试"。

实战标定会用好标品+足曝光（多环清晰）；全随机抽样则混入 1-2 环烂工况。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset
from torchfai.phases import PHASE_NAMES, phase_qi
from torchfai.simulator import ring_radius_m

cfg = json.load(open("data/train_v2/meta.json", encoding="utf-8"))["config"]
PIX = 172.0 * cfg["binning"] * 1e-6
Hm, Wm = 521 * PIX, 490 * PIX
lam_A = cfg["wavelength"] * 1e10
BS = (max(cfg["beamstop_r_range"]) + 3) * PIX

ds = TorchFAIDataset("data/train_v2", split="test")
qcache = {}


def nrings(ph, p):
    if ph not in qcache:
        qcache[ph] = phase_qi(ph)[0]
    rr = ring_radius_m(qcache[ph], p[0], lam_A)
    rf = 1.05 * float(np.hypot(0.75 * Hm + abs(p[1] - Hm / 2),
                               0.75 * Wm + abs(p[2] - Wm / 2)))
    return int(((rr > BS) & (rr < rf)).sum())


rows = [l.strip().split(",") for l in open("runs/refine_v2/results.csv",
                                           encoding="utf-8").readlines()[1:] if l.strip()]
ns, ea = [], []
for r in rows:
    k = int(r[0])
    g = int(ds._sel[k])
    ns.append(nrings(PHASE_NAMES[int(ds._maux[g][0])], ds._mparams[g].astype(np.float64)))
    ea.append(float(r[9]))
ns, ea = np.array(ns), np.array(ea)

print("按可见环数分层的卡住率（dist>5mm）：")
print(f"{'环数':>6s} {'样本':>6s} {'卡住率':>7s} {'中位误差mm':>10s}")
for lo, hi in ((0, 2), (3, 3), (4, 5), (6, 99)):
    m = (ns >= lo) & (ns <= hi)
    if m.sum() == 0:
        continue
    print(f"{lo}-{min(hi, 99):2d}  {m.sum():6d} {(ea[m] > 5).mean() * 100:6.1f}% "
          f"{np.median(ea[m]):10.3f}")

m4 = ns >= 4
print(f"\n实战工况(环>=4): N={m4.sum()} ({m4.mean() * 100:.0f}%)  "
      f"卡住率 {(ea[m4] > 5).mean() * 100:.1f}%  中位 {np.median(ea[m4]):.3f}mm")
print(f"全随机压力测试:   N={len(ns)}       "
      f"卡住率 {(ea > 5).mean() * 100:.1f}%  中位 {np.median(ea):.3f}mm")
