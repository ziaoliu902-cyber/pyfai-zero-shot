# -*- coding: utf-8 -*-
"""poni/rot 精修优化消融：独立学习率(A) + σ真值区间(B)。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 40
CONFIGS = {
    "baseline": dict(),
    "A:lr×3(poni,rot)": dict(lr_mult=(1, 3, 3, 3, 3, 1)),
    "B:σ真值区间": dict(sigma_range=(0.003, 0.011)),
    "A+B": dict(lr_mult=(1, 3, 3, 3, 3, 1), sigma_range=(0.003, 0.011)),
}

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

rng = np.random.default_rng(7)
ks = sorted(rng.choice(len(ds._sel), N, replace=False).tolist())
data = []
refs = {}
for k in ks:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    true_p = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
        mu, lv = model(x)
        pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
        sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
    data.append((img, true_p, ph, pred, sig))

names = ("dist", "poni1", "poni2", "rot1", "rot2")
print(f"{'配置':18s}" + "".join(f"{nm:>10s}" for nm in names) + "   (mm/mrad MAE)")
err0 = np.array([np.abs(p - t) * 1000 for _, t, _, p, _ in data])
print(f"{'NN粗解':18s}" + "".join(f"{err0[:, i].mean():10.3f}" for i in range(5)))
for tag, kw in CONFIGS.items():
    errs = []
    for img, true_p, ph, pred, sig in data:
        r = refs[ph].refine(img, pred, iters=400, sigma_norm=sig,
                            trust_k=0.5, **kw)
        errs.append(np.abs(r["params"] - true_p) * 1000)
    errs = np.array(errs)
    print(f"{tag:18s}" + "".join(f"{errs[:, i].mean():10.3f}" for i in range(5)))
