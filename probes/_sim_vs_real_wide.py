# -*- coding: utf-8 -*-
"""对照实验：sim 样本(与真实 400mm 帧同 dist/k 邻域)的粗解误差 vs 真实帧。
sim 准而真实偏 → 现实差(增益/挡光/饱和)；sim 也偏 → 模型在该区有洞。"""
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
from torchfai.dataset import TorchFAIDataset

m, dev, consts = load_px_nn(r"runs\v3far4\last.pt")
m.eval()
ds = TorchFAIDataset("data/train_v3x", "val")

# 找与真实宽帧同邻域的 sim 样本: k∈[2.3,2.55], dist∈[0.13,0.46]
import csv
idx = []
n = len(ds)
aux_all = np.load(r"data\train_v3x\aux.npy", mmap_mode="r")
P_all = np.load(r"data\train_v3x\params.npy", mmap_mode="r")
# val split 与 train 同源顺序（TorchFAIDataset 内部 seed 划分）——直接扫全部
cand = []
for i in range(0, n, 7):            # 抽样扫
    k = float(aux_all[i, 6]) if aux_all.shape[1] > 6 else 1.0
    d = float(P_all[i, 0])
    if 2.3 <= k <= 2.55 and 0.13 <= d <= 0.46:
        cand.append(i)
print(f"同邻域候选(sim): {len(cand)} 个（抽样步长7）")

errs, taken = [], 0
for i in cand[:40]:
    try:
        img, y, a = ds[i]
        img = img.numpy() if isinstance(img, torch.Tensor) else img
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        a = a.numpy() if isinstance(a, torch.Tensor) else a
        p_nn, _ = nn_coarse(m, dev, consts, img, float(a[5]), float(a[6]))
        errs.append((p_nn[0] - float(P_all[i, 0])) * 1000)
        taken += 1
    except Exception:
        continue
errs = np.array(errs)
if taken:
    print(f"sim 同邻域粗解误差(n={taken}): 均值={errs.mean():+.1f}mm "
          f"中位={np.median(errs):+.1f}mm 范围[{errs.min():+.0f},"
          f"{errs.max():+.0f}]")
print("\n真实帧粗解(v7考卷): 150:-2.9 200:-25.9 250:-37.1 300:-49.5 "
      "350:-68.5 400:-45.4 450:-51.5")
