# -*- coding: utf-8 -*-
"""检查 far_patch_v1 的 dist 分布/相构成，确认它能否当"远距回放"源。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset
from torchfai.phases import PHASE_NAMES

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds = TorchFAIDataset(os.path.join(ROOT, "data", "far_patch_v1"), split="train")
d = ds._mparams[ds._sel][:, 0]
aux = ds._maux[ds._sel]
print(f"far_patch_v1 N={len(ds._sel)}")
for lo, hi, nm in ((0.0, 0.77, "<0.77m"), (0.77, 1.0, "0.77-1.0m"),
                   (1.0, 2.0, "1.0-2.0m"), (2.0, 3.1, ">2.0m")):
    m = (d >= lo) & (d < hi)
    print(f"  {nm:10s} {m.mean()*100:5.1f}%  (N={m.sum()})")
print(f"  >0.77m {(d>0.77).mean()*100:.1f}%   >1.0m {(d>1.0).mean()*100:.1f}%")
ph = np.bincount(aux[:, 0].astype(int), minlength=6)
print("  相构成: " + "  ".join(f"{PHASE_NAMES[i]} {ph[i]/len(aux)*100:.0f}%" for i in range(6)))
prac = (aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
print(f"  实际工况画像比例: {prac.mean()*100:.1f}%")

# practice 段的"实际工况画像"细分：与 train_v2 test 的实际工况档对比
ft = TorchFAIDataset(os.path.join(ROOT, "data", "finetune_v1"), split="train")
af = ft._maux[ft._sel]
pm = (af[:, 1] >= 6) & (af[:, 3] >= 10000) & (af[:, 4] < 0)
rows_prac = np.arange(0, 12000)
m1 = pm & (ft._sel < 12000)
m2 = pm & (ft._sel >= 12000)
print(f"\nfinetune_v1 中符合档位定义: practice段 {m1.sum()} (占该段 "
      f"{m1.sum()/12000*100:.1f}%)  regular段 {m2.sum()} (占该段 {m2.sum()/8000*100:.1f}%)")
print(f"  practice 段内 dist: median={np.median(ft._mparams[ft._sel[m1], 0]):.3f}")
