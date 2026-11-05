# -*- coding: utf-8 -*-
"""审计 finetune_v1 的归一化标签是否越界 —— merge_ft.py 根因的代码级证据。

根因：merge_ft.py 用 packs[0]（practice 段）的 config 写 meta.json，而 practice 段的
dist_range=[0.1,1.0]；regular 段实际 dist 到 2.99m。于是：
  1) 微调集的 dist 标签编码空间与 base_v3（train_v3, dist_range=[0.1,3.0]）错开
     → init_from base_v3 后，dist 头被要求输出到"另一把尺子"上；
  2) dist>1.0m 的行归一化后 y>1，越出 [-1,1] 目标域。

本脚本用 finetune_v1 自己的 meta.json 常数实算越界数，并与 train_v3 的常数对照。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, param_to_norm

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ft = TorchFAIDataset(os.path.join(ROOT, "data", "finetune_v1"), split="train")
v3 = TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split="train")

p_ft = np.asarray(ft._mparams)
p_v3 = np.asarray(v3._mparams)

print("=" * 78)
print("归一化常数对照")
print("=" * 78)
for nm, c in (("finetune_v1", ft.norm_consts), ("train_v3 (base_v3 训练集)", v3.norm_consts)):
    print(f"  {nm:26s} dlo={c['dlo']:+.4f} dhi={c['dhi']:+.4f}"
          f"  → dist∈[10^{c['dlo']:.3f}, 10^{c['dhi']:.3f}]m")
print(f"  两套常数 dhi 差 {ft.norm_consts['dhi'] - v3.norm_consts['dhi']:+.4f}"
      f"  dlo 差 {ft.norm_consts['dlo'] - v3.norm_consts['dlo']:+.4f}")

print()
print("=" * 78)
print("标签越界审计（y 必须落在 [-1,1]）")
print("=" * 78)
for nm, ds, p in (("finetune_v1", ft, p_ft), ("train_v3", v3, p_v3)):
    y = param_to_norm(p, ds.norm_consts)
    bad = np.abs(y) > 1.0
    nd = bad[:, 0]
    print(f"\n  {nm}: N={len(p)}")
    print(f"    dist 越界行 {nd.sum():6d} ({nd.mean()*100:5.2f}%)"
          f"   y_dist 最大 {y[:, 0].max():+.4f}  最小 {y[:, 0].min():+.4f}")
    if nd.sum():
        d = p[nd, 0]
        print(f"    越界行的物理 dist: min={d.min():.3f}m  max={d.max():.3f}m")
        # 这些行来自哪个来源段
        for lo, hi, tag in ((0, 12000, "practice"), (12000, 20000, "regular")):
            if nm != "finetune_v1":
                break
            sub = nd[lo:hi]
            print(f"      其中 {tag:9s} 段 {sub.sum():5d} 行")

print()
print("=" * 78)
print("结论")
print("=" * 78)
y_ft = param_to_norm(p_ft, ft.norm_consts)
y_v3const = param_to_norm(p_ft, v3.norm_consts)
print(f"  同一批 finetune_v1 样本：")
print(f"    用 finetune_v1 自身常数归一化 → 越界 {int((np.abs(y_ft[:,0])>1).sum())} 行")
print(f"    用 train_v3 常数归一化        → 越界 {int((np.abs(y_v3const[:,0])>1).sum())} 行")
print(f"    同一批数据、同一批标签，仅常数不同 → 目标域与 base_v3 错开 = 根因")

# 直接量化"用错常数解码"的后果
from torchfai.dataset import norm_to_param
for nm, c in (("finetune_v1 常数", ft.norm_consts), ("train_v3 常数", v3.norm_consts)):
    dec = norm_to_param(y_v3const, c)[:, 0]
    err = np.abs(dec - p_ft[:, 0]) * 1000
    print(f"\n  若模型按 train_v3 空间输出、用【{nm}】解码：")
    print(f"    dist 误差 中位 {np.median(err):10.2f} mm   平均 {err.mean():10.2f} mm")
