# -*- coding: utf-8 -*-
"""poni 损失景观判决：真值附近扫 poni1，看最低点在哪、谷有多深。"""
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
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

g = int(ds._sel[K])
img = np.asarray(ds._mimgs[g])
true_p = ds._mparams[g].astype(np.float64)
ph = PHASE_NAMES[int(ds._maux[g][0])]
with torch.no_grad():
    x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
    mu, lv = model(x)
    pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
    sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
print(f"#{K} {ph} dist={true_p[0]:.3f}m n_vis={ds._maux[g][1]:.0f}")
print(f"NN: poni1 err={abs(pred[1]-true_p[1])*1000:.2f}mm  "
      f"rot1 err={abs(pred[3]-true_p[3])*1000:.2f}mrad")

ref = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))

print("\n--- poni1 扫描（其余=真值, ±4mm）---")
for off in (-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4):
    p = true_p.copy(); p[1] += off / 1000.0
    r = ref.refine(img, p, iters=25, warmup=120, trust_k=0.0)
    print(f"  off={off:+5.1f}mm  loss={r['loss']:12.4f}")

print("\n--- rot1 扫描（其余=真值, ±4mrad）---")
for off in (-4, -2, -1, 0, 1, 2, 4):
    p = true_p.copy(); p[3] += off / 1000.0
    r = ref.refine(img, p, iters=25, warmup=120, trust_k=0.0)
    print(f"  off={off:+5.1f}mrad loss={r['loss']:12.4f}")
