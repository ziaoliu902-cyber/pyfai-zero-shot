# -*- coding: utf-8 -*-
"""判决实验：从真值/单参数打偏出发的联合优化行为。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
g = int(ds._sel[K])
img = np.asarray(ds._mimgs[g])
tp = ds._mparams[g].astype(np.float64)
ph = PHASE_NAMES[int(ds._maux[g][0])]
ref = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
print(f"#{K} {ph} dist={tp[0]:.3f}m  trust 全开 k=4, iters=400")

def show(tag, start):
    r = ref.refine(img, start, iters=400, trust_k=4.0)
    e = (r["params"] - tp) * 1000
    print(f"{tag:26s} 终误差: dist={e[0]:+7.2f}mm poni1={e[1]:+7.2f}mm "
          f"poni2={e[2]:+6.2f}mm rot1={e[3]:+6.2f} rot2={e[4]:+6.2f} mrad  loss={r['loss']:.2f}")

show("A: 从真值出发", tp)
p = tp.copy(); p[1] += 0.004
show("B: 真值+poni1偏4mm", p)
p = tp.copy(); p[3] += 0.024
show("C: 真值+rot1偏24mrad", p)
p = tp.copy(); p[1] += 0.002; p[2] -= 0.002; p[0] += 0.005
show("D: 真值+组合偏(5mm,±2mm)", p)
