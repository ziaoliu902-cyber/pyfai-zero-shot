# -*- coding: utf-8 -*-
"""v2 vs v3 分档粗解评测（同卷 train_v2 test，3900 张）。

档位：全体 / 实际工况(n_vis≥6 & imax≥10k & 无驼峰) / 远距(>0.77m)。
指标：dist MAE+中位、粗解尾部(>10mm)、poni px 中位、rot 中位、
      σ 校准比(实际|err|中位 / σ中位，≈1 为佳)。
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.phases import PHASE_NAMES

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
aux = ds._maux[ds._sel]
nvis, imax, hump = aux[:, 1], aux[:, 3], aux[:, 4]
prac = (nvis >= 6) & (imax >= 10000) & (hump < 0)
dists = ds._mparams[ds._sel][:, 0]
far = dists > 0.77
PX_UM = 344.0    # binning 像素(μm)，与先前探针一致

TIERS = [("全体", np.ones(len(ds._sel), bool)),
         ("实际工况", prac),
         ("远距>0.77m", far)]

CKPTS = [("base_v2", os.path.join(ROOT, "runs/base_v2/best.pt")),
         ("base_v3", os.path.join(ROOT, "runs/base_v3/best.pt")),
         ("base_v4b", os.path.join(ROOT, "runs/base_v4b/best.pt"))]

for tag, fp in CKPTS:
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval().cuda()
    E = np.zeros((len(ds._sel), 6))       # 物理误差 (m, m, rad)
    S = np.zeros((len(ds._sel), 6))       # σ 物理化
    t0 = time.time()
    B = 64
    idx_all = np.arange(len(ds._sel))
    for s in range(0, len(idx_all), B):
        ks = idx_all[s:s + B]
        xb = np.stack([np.log1p(np.asarray(ds._mimgs[int(ds._sel[k])],
                                           dtype=np.float32)) for k in ks])
        with torch.no_grad():
            x = torch.from_numpy(xb)[:, None].cuda()
            mu, lv = model(x)
        for j, k in enumerate(ks):
            g = int(ds._sel[k])
            E[k] = np.abs(norm_to_param(mu[j].cpu().numpy().astype(np.float64),
                                        ds.norm_consts) - ds._mparams[g])
            sig_n = np.exp(0.5 * lv[j].cpu().numpy())
            S[k, 0] = sig_n[0] * ds._mparams[g][0] * 1.7005   # σ_dist→m
    print(f"\n===== {tag}（{time.time()-t0:.0f}s）=====")
    print(f"{'档':10s}{'N':>5s}{'distMAE':>9s}{'dist中位':>9s}{'尾>10mm':>8s}"
          f"{'poni中位px':>10s}{'rot1中位mrad':>12s}{'σ校准比':>8s}")
    for tname, m in TIERS:
        e = E[m] * 1000
        poni_px = np.median((e[:, 1] + e[:, 2]) / 2 / PX_UM * 1000)
        cal = np.median(np.abs(E[m][:, 0]) * 1000 / np.maximum(S[m][:, 0] * 1000, 1e-9))
        print(f"{tname:10s}{m.sum():5d}{np.mean(e[:, 0]):9.2f}"
              f"{np.median(e[:, 0]):9.3f}{(e[:, 0] > 10).mean()*100:7.1f}%"
              f"{poni_px:10.2f}{np.median(e[:, 3]):12.3f}{cal:8.2f}")
