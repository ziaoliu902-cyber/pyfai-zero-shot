# -*- coding: utf-8 -*-
"""钉子户靶向复测：已知精修失败的样本，测余弦衰减是否修复。"""
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
BAD = [0, 29, 41, 94, 119, 121]          # 上一轮发散样本（同一洗牌序）
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 400

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

refs = {}
for k in BAD:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    true_p = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
        mu, lv = model(x)
        pred_phys = norm_to_param(mu[0].cpu().numpy().astype(np.float64),
                                  ds.norm_consts)
        sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
    r = refs[ph].refine(img, pred_phys, iters=ITERS, sigma_norm=sig, trust_k=1.0)
    e0 = abs(pred_phys[0] - true_p[0]) * 1000
    e1 = abs(r["params"][0] - true_p[0]) * 1000
    sig_mm = sig[0] / 2.0 * (np.log10(3.0) - np.log10(0.1)) / np.log10(np.e) * true_p[0] * 1000  # σ_dist 归一化→mm 近似
    print(f"#{k:3d} {ph:7s} dist={true_p[0]:.2f}m  粗解{e0:7.2f} → 精修{e1:7.2f}mm  "
          f"({'FIXED' if e1 < max(2*e0, 1.0) else 'still bad'})  "
          f"σ_dist≈{sig_mm:.0f}mm  loss {r['loss_init']:.1f}→{r['loss']:.1f}")
