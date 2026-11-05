# -*- coding: utf-8 -*-
"""对照实验 v2（修索引bug）：真值取自 ds 的 y 反归一化。
sim 同邻域粗解误差 vs 真实帧 → 判现实差还是模型洞。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
from real_refine_v6 import load_px_nn, nn_coarse
from torchfai.dataset import TorchFAIDataset, norm_to_param

m, dev, consts = load_px_nn(r"runs\v3far4\last.pt")
m.eval()
ds = TorchFAIDataset("data/train_v3x", "val")
print(f"val 卷: {len(ds)} 个")

cand = []
for i in range(len(ds)):
    img, y, a = ds[i]
    a_ = a.numpy() if isinstance(a, torch.Tensor) else a
    y_ = y.numpy() if isinstance(y, torch.Tensor) else y
    k = float(a_[6]) if a_.shape[-1] > 6 else 1.0
    d_true = float(norm_to_param(y_[None], consts, k)[0, 0])
    if 2.3 <= k <= 2.55 and 0.13 <= d_true <= 0.46:
        cand.append((i, k, d_true))
print(f"同邻域(k2.3~2.55, d0.13~0.46): {len(cand)} 个")

errs = []
for i, k, d_true in cand[:40]:
    img, y, a = ds[i]
    img = img.numpy() if isinstance(img, torch.Tensor) else img
    a_ = a.numpy() if isinstance(a, torch.Tensor) else a
    p_nn, _ = nn_coarse(m, dev, consts, img, float(a_[5]), k)
    errs.append((p_nn[0] - d_true) * 1000)
errs = np.array(errs)
print(f"sim 同邻域粗解误差(n={len(errs)}): "
      f"均值={errs.mean():+.1f} 中位={np.median(errs):+.1f} "
      f"范围[{errs.min():+.0f},{errs.max():+.0f}]mm")
print("\n真实帧粗解: 150:-2.9 200:-25.9 250:-37.1 300:-49.5 "
      "350:-68.5 400:-45.4 450:-51.5")
print("\n判读: sim 准(±20mm内)而真实偏(-30~-70) → 现实差；"
      "sim 也偏 → 该 (dist,k) 角落模型本身没学会")
