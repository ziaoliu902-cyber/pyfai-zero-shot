# -*- coding: utf-8 -*-
"""模型地板测量：val 同邻域（k∈[2.3,2.55] × 各距离档）带符号 dist 偏差。
若仿真内邻域就有 -30~-60mm 负偏 → 病在模型/数据密度，不在真实帧的皮；
若邻域干净 → 皮才是主罪（再回头撕皮）。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from torchfai.dataset import TorchFAIDataset, norm_to_param

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v4w2"), "val")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(os.path.join(ROOT, "runs", "v4w2", "best.pt"),
                map_location="cpu", weights_only=False)
from torchfai.model import ResNet18Geo
m = ResNet18Geo(lam_cond=True, px_cond=True).eval().to(dev)
m.load_state_dict(ck["model"])
consts = ds.norm_consts

BINS = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.55),
        (0.55, 1.00), (1.00, 3.01)]
res = {b: [] for b in BINS}
res_narrow = {b: [] for b in BINS}          # 同距离但 k<2（对照）

N = len(ds)
for i in range(N):
    img, y, a = ds[i]
    a_ = a.numpy() if hasattr(a, "numpy") else a
    y_ = y.numpy().astype(np.float64) if hasattr(y, "numpy") else y
    k = float(a_[6]) if a_.shape[-1] > 6 else 1.0
    lam = float(a_[5]) if a_.shape[-1] > 5 else 1.5406
    d_true = float(norm_to_param(y_[None].astype(np.float64), consts, k)[0, 0])
    if not (0.7 < lam < 1.3):
        continue
    for b in BINS:
        if b[0] <= d_true < b[1]:
            tgt = res if 2.3 <= k <= 2.55 else (
                res_narrow if k < 2.0 else None)
            if tgt is None:
                break
            x = img[None, None]                     # dataset 已 log1p
            if not torch.is_tensor(x):
                x = torch.from_numpy(np.asarray(x, np.float32))
            x = x.float().to(dev)
            with torch.no_grad():
                mu, _ = m(x, torch.tensor([lam], device=dev),
                          torch.tensor([k], device=dev))
            d_pred = float(norm_to_param(
                mu.double().cpu().numpy(), consts, k)[0, 0])
            tgt[b].append((d_pred - d_true) * 1000)
            break

print(f"{'距离档m':>10s}{'k2.3-2.55 n':>13s}{'带符号偏差mm':>13s}"
      f"{'|err|中位':>10s}{'k<2 n':>8s}{'对照偏差':>10s}")
for b in BINS:
    r, rn = np.array(res[b]), np.array(res_narrow[b])
    if len(r) == 0:
        print(f"{str(b):>10s}{'0':>13s}{'-':>13s}{'-':>10s}"
              f"{len(rn):>8d}"
              f"{(rn.mean() if len(rn) else float('nan')):+10.1f}")
        continue
    print(f"{str(b):>10s}{len(r):>13d}{r.mean():+13.1f}"
          f"{np.median(np.abs(r)):10.1f}{len(rn):>8d}"
          f"{(rn.mean() if len(rn) else float('nan')):+10.1f}")
