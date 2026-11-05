# -*- coding: utf-8 -*-
"""对照实验 v3：不走 nn_coarse（它会再 log1p 一次），直接前向。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
from real_refine_v6 import load_px_nn
from torchfai.dataset import TorchFAIDataset, norm_to_param

m, dev, consts = load_px_nn(r"runs\v3far4\last.pt")
m.eval()
ds = TorchFAIDataset("data/train_v3x", "val")
print(f"val 卷: {len(ds)} 个")

cand = []
for i in range(len(ds)):
    _, y, a = ds[i]
    y_ = y.numpy() if isinstance(y, torch.Tensor) else y
    a_ = a.numpy() if isinstance(a, torch.Tensor) else a
    k = float(a_[6]) if a_.shape[-1] > 6 else 1.0
    d_true = float(norm_to_param(y_[None].astype(np.float64), consts, k)[0, 0])
    if 2.3 <= k <= 2.55 and 0.13 <= d_true <= 0.46:
        cand.append((i, k, d_true))
print(f"同邻域(k2.3~2.55, d0.13~0.46): {len(cand)} 个")

errs, poni_errs = [], []
for i, k, d_true in cand[:60]:
    img, y, a = ds[i]
    img = img.numpy() if isinstance(img, torch.Tensor) else img
    y_ = y.numpy().astype(np.float64)
    a_ = a.numpy() if isinstance(a, torch.Tensor) else a
    x = torch.from_numpy(img[None, None].astype(np.float32)).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([float(a_[5])], device=dev),
                  torch.tensor([k], device=dev))
    p = norm_to_param(mu.double().cpu().numpy(), consts, k)[0]
    errs.append((p[0] - d_true) * 1000)
    poni_errs.append((p[1] - norm_to_param(y_[None], consts, k)[0, 1])
                     / (consts["Hm"] * k / 521))   # 粗略换 px
errs = np.array(errs)
print(f"sim 同邻域粗解误差(n={len(errs)}): "
      f"均值={errs.mean():+.1f} 中位={np.median(errs):+.1f} "
      f"范围[{errs.min():+.0f},{errs.max():+.0f}]mm")
print("真实帧粗解: 150:-2.9 200:-25.9 250:-37.1 300:-49.5 "
      "350:-68.5 400:-45.4 450:-51.5")
