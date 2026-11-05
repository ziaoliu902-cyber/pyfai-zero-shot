# -*- coding: utf-8 -*-
"""统计 finetune_v1 训练切分的 dist 配比，为重平衡微调集提供依据。

同时澄清 aux 列真实语义（simulator.py 第 220 行）：
    aux = [phase_idx, n_vis, qmax, scale, bs_r]
  → col3 是 scale（绝对标度系数），不是"峰值计数 imax"
  → col4 是 bs_r（挡光盘半径，-1 = 无挡光盘），不是"驼峰幅度"
历史脚本把 col3/col4 命名为 imax/hump 是错的（注释误导，数值本身自洽）。
本探针只统计，不改数据。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ft = TorchFAIDataset(os.path.join(ROOT, "data", "finetune_v1"), split="train")
sel = ft._sel
aux = ft._maux[sel]
par = ft._mparams[sel]

nvis, scale, bsr = aux[:, 1], aux[:, 3], aux[:, 4]
prac_img = (nvis >= 6) & (scale >= 10000) & (bsr < 0)     # 档位定义(原样)
d = par[:, 0]

print(f"finetune_v1 train: N={len(sel)}")
print(f"\n--- dist 配比（train 切分）---")
for lo, hi, nm in ((0.0, 0.35, "0.10-0.35m 近距"),
                   (0.35, 0.77, "0.35-0.77m 中距"),
                   (0.77, 1.0, "0.77-1.00m 远"),
                   (1.0, 3.1, ">1.00m 极远")):
    m = (d >= lo) & (d < hi)
    print(f"  {nm:16s} N={m.sum():6d} ({m.mean()*100:5.1f}%)  "
          f"其中实际工况画像 {prac_img[m].sum():5d} ({prac_img[m].mean()*100 if m.sum() else 0:5.1f}%)")
print(f"  远距(>0.77m) 合计 {(d>0.77).mean()*100:5.1f}%   "
      f">1.0m {(d>1.0).mean()*100:5.1f}%")
print(f"  实际工况画像合计 {prac_img.mean()*100:5.1f}%  (N={prac_img.sum()})")

print("\n--- 对比 train_v3（v3 的训练集，dist 已知是平衡的）---")
v3 = TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split="train")
s3 = v3._sel
a3 = v3._maux[s3]
d3 = v3._mparams[s3][:, 0]
p3 = (a3[:, 1] >= 6) & (a3[:, 3] >= 10000) & (a3[:, 4] < 0)
print(f"  train_v3 N={len(s3)}")
for lo, hi, nm in ((0.0, 0.35, "0.10-0.35m"), (0.35, 0.77, "0.35-0.77m"),
                   (0.77, 1.0, "0.77-1.00m"), (1.0, 3.1, ">1.00m")):
    m = (d3 >= lo) & (d3 < hi)
    print(f"    {nm:12s} {m.mean()*100:5.1f}%")
print(f"  远距(>0.77m) {(d3>0.77).mean()*100:5.1f}%   >1.0m {(d3>1.0).mean()*100:5.1f}%")
print(f"  实际工况画像 {p3.mean()*100:5.1f}%  (N={p3.sum()})")

print("\n--- aux 列语义澄清（对照 simulator.py:220）---")
print(f"  col3=scale : min={scale.min():.1f} max={scale.max():.1f} "
      f"median={np.median(scale):.1f}   ← 绝对标度系数 target/ring_max_vis")
print(f"  col4=bs_r  : min={bsr.min():.1f} max={bsr.max():.1f} "
      f"median={np.median(bsr):.1f}  <0 的比例 {(bsr<0).mean()*100:.1f}%  ← 无挡光盘")
print(f"  驼峰幅度根本没进 aux（hump 只影响生成，不记录）")
