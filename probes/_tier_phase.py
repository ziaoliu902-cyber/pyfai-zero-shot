# -*- coding: utf-8 -*-
"""核对"实际工况档"的相构成 vs practice 段的相构成 —— 确认微调集是否打到了正确的靶。

档位定义（沿用历史口径）：nvis>=6 & scale>=10000 & bs_r<0（无挡光盘）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset
from torchfai.phases import PHASE_NAMES

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

v2 = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
a = v2._maux[v2._sel]
prac = (a[:, 1] >= 6) & (a[:, 3] >= 10000) & (a[:, 4] < 0)
print("train_v2 test 各相占比 vs 实际工况档内各相占比：")
print(f"{'相':10s}{'全体%':>9s}{'档内%':>9s}{'档内N':>8s}")
ph_all = np.bincount(a[:, 0].astype(int), minlength=6)
ph_prac = np.bincount(a[prac, 0].astype(int), minlength=6)
for i, nm in enumerate(PHASE_NAMES):
    print(f"{nm:10s}{ph_all[i]/len(a)*100:9.1f}{ph_prac[i]/max(ph_prac.sum(),1)*100:9.1f}"
          f"{ph_prac[i]:8d}")
print(f"\n档内 N={prac.sum()}")

# practice 段（微调集主体）的相构成
ft = TorchFAIDataset(os.path.join(ROOT, "data", "finetune_v1"), split="train")
af = ft._maux[ft._sel]
print("\nfinetune_v1 train：")
for lo, hi, nm in ((0, 12000, "practice 段"), (12000, 20000, "regular 段")):
    m = (ft._sel >= lo) & (ft._sel < hi)
    ph = np.bincount(af[m, 0].astype(int), minlength=6)
    print(f"  {nm} (N={m.sum()}): " +
          "  ".join(f"{PHASE_NAMES[i]} {ph[i]/m.sum()*100:.0f}%" for i in range(6)))
