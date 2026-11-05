# -*- coding: utf-8 -*-
"""软 MAP 先验 A/B：实际工况档 + 全体档，各 30 张。

C0  = 现行配方（硬笼 0.5σ）
M1/2/3 = 硬笼放宽 1σ 当保险丝 + 软先验 w=0.1/1/10
判据：实际工况档 poni 3.34px / rot2 7.7mrad 应大幅下降；
      全体档 dist 中位 0.07mm / 卡住率不得恶化。
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

CONFIGS = [
    ("C0 硬笼0.5σ", dict(trust_k=0.5)),
    ("M1 软w=0.1", dict(trust_k=1.0, soft_prior=0.1)),
    ("M2 软w=1", dict(trust_k=1.0, soft_prior=1.0)),
    ("M3 软w=10", dict(trust_k=1.0, soft_prior=10.0)),
]


def load_samples(sel):
    out, refs = [], {}
    for k in sel:
        g = int(ds._sel[k])
        img = np.asarray(ds._mimgs[g])
        tp = ds._mparams[g].astype(np.float64)
        ph = PHASE_NAMES[int(ds._maux[g][0])]
        with torch.no_grad():
            x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
            mu, lv = model(x)
            pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64),
                                 ds.norm_consts)
            sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
        if ph not in refs:
            refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
        out.append((img, tp, ph, pred, sig))
    return out, refs


aux = ds._maux[ds._sel]
prac_idx = np.where((aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0))[0]
rng_p = np.random.default_rng(11)
sel_p = sorted(rng_p.choice(prac_idx, 30, replace=False).tolist())
rng_a = np.random.default_rng(7)
sel_a = sorted(rng_a.choice(len(ds._sel), 30, replace=False).tolist())

P, refs_p = load_samples(sel_p)
A, refs_a = load_samples(sel_a)
PX_UM = refs_p[P[0][2]].db.pixel * 1e6
names = ("dist", "poni1", "poni2", "rot1", "rot2")

for tier, data, refs in (("实际工况档(30)", P, refs_p), ("全体档(30)", A, refs_a)):
    print(f"\n===== {tier}  像素{PX_UM:.0f}μm =====", flush=True)
    print(f"{'配置':14s}{'dist中位':>9s}{'<0.5mm':>7s}{'poni中位px':>10s}"
          f"{'rot1中位':>9s}{'rot2中位':>9s}{'卡住':>5s}")
    for tag, kw in CONFIGS:
        t0 = time.time()
        errs = []
        for img, tp, ph, pred, sig in data:
            r = refs[ph].refine(img, pred, iters=400, sigma_norm=sig, **kw)
            errs.append(np.abs(r["params"] - tp) * 1000)
        e = np.array(errs)
        poni_px = np.median((e[:, 1] + e[:, 2]) / 2 / PX_UM * 1000)
        print(f"{tag:14s}{np.median(e[:, 0]):9.3f}"
              f"{(e[:, 0] < 0.5).mean()*100:6.0f}%{poni_px:10.2f}"
              f"{np.median(e[:, 3]):9.3f}{np.median(e[:, 4]):9.3f}"
              f"{(e[:, 0] > 5).mean()*100:4.0f}%  {time.time()-t0:.0f}s",
              flush=True)
