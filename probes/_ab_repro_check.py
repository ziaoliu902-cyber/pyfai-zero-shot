# -*- coding: utf-8 -*-
"""复现 A/B 的抽样与精修，逐样本对照 NPZ —— 判定差异来自抽样噪声还是口径 bug。"""
import glob
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
RES = os.path.join(ROOT, "probes", "results")

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"), map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

aux = ds._maux[ds._sel]
prac_idx = np.where((aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0))[0]
rng_p = np.random.default_rng(11)
sel_p = sorted(rng_p.choice(prac_idx, 30, replace=False).tolist())
print(f"pool={len(prac_idx)}  抽中 30 张（与 A/B 同参）")

refs = {}
errs = []
for k in sel_p:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    tp = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
        mu, lv = model(x)
        pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
        sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
    r = refs[ph].refine(img, pred, iters=400, sigma_norm=sig, trust_k=0.5)
    errs.append(np.abs(r["params"] - tp) * 1000)
e = np.array(errs)
px = (e[:, 1] + e[:, 2]) / 2 / 0.344
print(f"复现: poni px 中位 {np.median(px):.2f}  dist 中位 {np.median(e[:,0]):.3f}mm  "
      f"<0.5mm {(e[:,0]<0.5).mean()*100:.0f}%")
print("逐样本 poni px:", np.round(px, 2))

npz_fp = sorted(glob.glob(os.path.join(RES, "map_prior_ab_*.npz")))[-1]
z = np.load(npz_fp)
key = [k for k in z.files if k.startswith("实际工况档") and "C0" in k][0]
ref = z[key]
ref_px = (ref[:, 1] + ref[:, 2]) / 2 / 0.344
print(f"\nA/B NPZ: poni px 中位 {np.median(ref_px):.2f}")
print("逐样本 poni px:", np.round(ref_px, 2))
print("\n逐样本最大绝对差(poni px):", np.abs(px - ref_px).max())
print("逐样本最大绝对差(dist mm):", np.abs(e[:, 0] - ref[:, 0]).max())
