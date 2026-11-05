# -*- coding: utf-8 -*-
"""poni 似然切片：纯曲率测量（无优化污染）。

每个偏移点：几何=真值(仅 poni1 加偏移)，nuisance 暖机 120 步后读 loss_init。
若 ±2mm 内 Δloss ≈ 0 → 结构性简并（nuisance 吸收）；若 Δloss 显著 →
信息在，是优化/先验问题。5 张实际工况样本。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
aux = ds._maux[ds._sel]
prac = np.where((aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0))[0]
sel = prac[np.linspace(0, len(prac) - 1, 5).astype(int)]   # 均匀取5张

OFFS = (-4, -2, -1, -0.5, 0.5, 1, 2, 4)        # mm
refs = {}
for j, k in enumerate(sel):
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    tp = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
    r0 = refs[ph].refine(img, tp, warmup=120, iters=0, sigma_norm=None)
    l0 = r0["loss_init"]
    row = []
    for off in OFFS:
        p = tp.copy(); p[1] += off / 1000.0
        r = refs[ph].refine(img, p, warmup=120, iters=0, sigma_norm=None)
        row.append(r["loss_init"] - l0)
    print(f"#{j} {ph:6s} dist={tp[0]:.2f}m n_vis={ds._maux[g][1]:.0f} "
          f"imax={ds._maux[g][3]/1000:.0f}k  loss0={l0:.4f}")
    print("  off(mm): " + " ".join(f"{o:+6.1f}" for o in OFFS))
    print("  Δloss:   " + " ".join(f"{d:+6.3f}" for d in row))

print("\n判读: Δloss>1 → 似然对 poni 敏感(优化问题); Δloss≈0 → 结构性简并")
