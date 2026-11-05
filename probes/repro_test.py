# -*- coding: utf-8 -*-
"""复现性测试：同一样本同参数连跑 5 次，检验精修是否确定性。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

ds = TorchFAIDataset("data/train_v2", split="test")
ck = torch.load("runs/base_v2/best.pt", map_location="cpu", weights_only=False)
m = ResNet18Geo()
m.load_state_dict(ck["model"] if "model" in ck else ck)
m.eval().cuda()

for K in (155, 7):
    g = int(ds._sel[K])
    img = np.asarray(ds._mimgs[g])
    tp = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        mu, lv = m(torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda())
        sg = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
        pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
    ref = DifferentiableRefiner(ph, "data/train_v2")
    print(f"--- #{K} {ph} 同参数连跑5次(1500迭代) ---")
    for i in range(5):
        r = ref.refine(img, pred, iters=1500, sigma_norm=sg, trust_k=0.5)
        e = np.abs(r["params"] - tp) * 1000
        print(f"  第{i+1}次: dist {e[0]:8.3f}mm  poni1 {e[1]:7.3f}mm  "
              f"rot1 {e[3]:6.2f}mrad  loss {r['loss']:.1f}")
