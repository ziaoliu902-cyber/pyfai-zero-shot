# -*- coding: utf-8 -*-
"""损失景观诊断：真值附近扫 dist，检查损失最低点位置。"""
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
K = int(sys.argv[1]) if len(sys.argv) > 1 else 119

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
    mu, _ = model(x)
    pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
print(f"#{K} {ph}  true dist={true_p[0]:.4f}m  NN pred={pred[0]:.4f}m "
      f"(err {abs(pred[0]-true_p[0])*1000:.1f}mm)  aux={ds._maux[g]}")

ref = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
print("\noffset(mm)   dist(m)    loss(冻结几何+暖机nuisance)")
for off in (-60, -40, -25, -12, -6, -3, 0, 3, 6, 12, 25, 40, 60):
    p = true_p.copy()
    p[0] += off / 1000.0
    r = ref.refine(img, p, iters=30, warmup=120)
    mark = " ← NN" if abs(p[0] - pred[0]) < 6e-3 else ""
    print(f"{off:+8.0f}  {p[0]:.4f}   {r['loss']:12.3f}{mark}")
