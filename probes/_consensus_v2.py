# -*- coding: utf-8 -*-
"""共识硬化 v2：修正选主规则 + poni 抖动加密 + σ 信任域。

v1 教训（runs/consensus.log）：
- 450mm 的 7 票紧簇在真值 +0.3mm，被 1 票假盆地按 minloss 抢走 →
  选主改为 score = minloss − 2·(size−1)，票数话语权优先
- 观测 GT-似然 poni 系统差 -16~-23px > 抖动 ±12px → 加密到 8 向 ±16px
- AgBh poni 跑飞 (+589,+515)px 进 73-nat 垃圾深坑 → 用 NN σ 信任域
  （refiner 自带 sigma_norm+trust_k，今日首次启用）
只重跑 v1 失败四档。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
import fabio
import pyFAI

from real_refine_v5 import to_nn_input_phys
from real_refine_v6 import (REAL, ROOT, load_px_nn, match_gain, nn_coarse,
                            to_nn_input_wide)
from torchfai.refine import DifferentiableRefiner

DIST_GRID = (-32, -16, 0, 16, 32, 48, 64)
PONI_JIT = [(0.0, 0.0), (16, 0), (-16, 0), (0, 16), (0, -16),
            (16, 16), (-16, -16), (16, -16), (-16, 16)]
TOL_MM = 20.0
VOTE_W = 2.0

m, dev, consts = load_px_nn(os.path.join(ROOT, "runs/px_v2/best.pt"))


def consensus2(finals):
    finals = sorted(finals, key=lambda f: f[0])
    clusters = []
    for f in finals:
        for c in clusters:
            if abs(f[1] - c["med"]) <= TOL_MM / 1000:
                c["members"].append(f)
                c["med"] = float(np.median([x[1] for x in c["members"]]))
                break
        else:
            clusters.append(dict(med=f[1], members=[f]))
    for c in clusters:
        c["minloss"] = min(x[0] for x in c["members"])
        c["score"] = c["minloss"] - VOTE_W * (len(c["members"]) - 1)
        c["iqr"] = (np.percentile([x[1] for x in c["members"]], 75) -
                    np.percentile([x[1] for x in c["members"]], 25)) * 1000
    clusters.sort(key=lambda c: c["score"])
    best = clusters[0]
    rep = min(best["members"], key=lambda x: x[0])
    diag = [(len(c["members"]), round(c["minloss"], 1), round(c["score"], 1),
             round((c["med"] - best["med"]) * 1000, 1), round(c["iqr"], 1))
            for c in clusters[:4]]
    return rep, diag


def campaign(ref, img, gt, d0, pon, px, use_trust=False, sig=None,
             dhi=1.0, lam_sig=None):
    finals = []
    for dm in DIST_GRID:
        st = np.array([np.clip(d0 + dm * 1e-3, 0.1, dhi),
                       pon[0], pon[1], 0, 0, 0])
        r = ref.refine(img, st, iters=400, sigma_norm=sig, trust_k=3.0,
                       sigma_range=(0.0015, 0.02))
        finals.append((r["loss"], r["params"][0], r["params"]))
    d1 = float(np.median([f[1] for f in sorted(finals)[:3]]))
    for jy, jx in PONI_JIT[1:]:
        st = np.array([d1, pon[0] + jy * px, pon[1] + jx * px, 0, 0, 0])
        r = ref.refine(img, st, iters=400, sigma_norm=sig, trust_k=3.0,
                       sigma_range=(0.0015, 0.02))
        finals.append((r["loss"], r["params"][0], r["params"]))
    rep, diag = consensus2(finals)
    ed = (rep[1] - gt[0]) * 1000
    print(f"共识 distΔ={ed:+7.2f}mm  poniΔ=({(rep[2][1]-gt[1])/px:+5.1f},"
          f"{(rep[2][2]-gt[2])/px:+5.1f})px  loss={rep[0]:8.2f}  "
          f"簇(票,minloss,score,偏距,IQR)={diag}", flush=True)


ref6 = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus6M", wavelength_m=9.72386e-11,
                             dist_range=[0.1, 1.0])
px = 172e-6
for d in (350, 400, 450):
    img = fabio.open(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    wide, k, extra = to_nn_input_wide(img)
    p_nn, sig_n = nn_coarse(m, dev, consts, wide, 0.972386, k)
    pon = np.array([p_nn[1], p_nn[2] + extra])
    print(f"[ceriaW {d}mm]", end=" ", flush=True)
    campaign(ref6, img, gt, p_nn[0], pon, px, sig=sig_n)

ref2 = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus_1M", wavelength_m=1.000e-10,
                             dist_range=[0.1, 3.0])
img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
p = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))
gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
p_nn, sig_n = nn_coarse(m, dev, consts,
                        match_gain(to_nn_input_phys(img)) * 0.5, 1.0, 1.0)
print("[AgBh 1M]", end=" ", flush=True)
campaign(ref2, img, gt, p_nn[0], np.array([p_nn[1], p_nn[2]]), px,
         sig=sig_n, dhi=3.0)
