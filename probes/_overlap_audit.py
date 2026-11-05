# -*- coding: utf-8 -*-
"""跨代切分重叠实测：base_v3 训练集 与 各评估集 的交集。

前提（已由 probes/audit_train_v3.py 验证）：train_v3 的前 65000 行与 train_v2
逐位恒等，故可用同一全局下标直接求交。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

v2 = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="train")
v2te = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
v2va = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="val")
v3 = TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split="train")
v3te = TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split="test")
v3va = TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split="val")

tr2 = set(v2._sel.tolist())
tr3 = set(v3._sel.tolist())
te3 = set(v3te._sel.tolist())
va3 = set(v3va._sel.tolist())

print(f"train_v2: train={len(tr2)} val={len(v2va._sel)} test={len(v2te._sel)}")
print(f"train_v3: train={len(tr3)} val={len(va3)} test={len(te3)}")
print()

def frac(name, evalset, trainset):
    n = len(evalset)
    k = len(evalset & trainset)
    print(f"{name:52s} {k:5d}/{n:5d} = {k/n*100:5.1f}%")


frac("base_v3 训练集 ∩ train_v2 test（评估 v3 在 v2 卷上）", set(v2te._sel.tolist()), tr3)
frac("base_v3 训练集 ∩ train_v2 val", set(v2va._sel.tolist()), tr3)
frac("base_v2 训练集 ∩ train_v3 test（评估 v2 在 v3 卷上）", te3, tr2)
frac("base_v3 训练集 ∩ train_v3 test（v3 自测，应为 0）", te3, tr3)
frac("base_v3 训练集 ∩ train_v3 val（v3 自选型集，应为 0）", va3, tr3)

# train_v3 test 中属于 v2 块(=下标<65000)的行
v3te_arr = v3te._sel
in_v2blk = v3te_arr[v3te_arr < 65000]
print()
print(f"train_v3 test 中属 v2 块(<65000)的行：{len(in_v2blk)}/{len(v3te_arr)}")
frac("  其中曾入 base_v2 训练集", set(in_v2blk.tolist()), tr2)
frac("  其中曾入 base_v2 val", set(in_v2blk.tolist()), set(v2va._sel.tolist()))
frac("  其中曾入 base_v2 test", set(in_v2blk.tolist()), set(v2te._sel.tolist()))
