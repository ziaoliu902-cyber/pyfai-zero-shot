# -*- coding: utf-8 -*-
"""诊断 finetune_v1 的 dist 归一化常数是否与实际数据范围一致。

背景：合并脚本 merge_ft.py 用 packs[0]（practice 段）的 meta.config 作为合并后
meta 的 config —— 它的 dist_range 被覆写成 [0.1, 1.0]；而 regular 段 8000 张
真实 dist 到 3.0m。torchfai.dataset._norm_consts 从 meta.config 取归一化常数，
于是 regular 段的 dist 标签被推到 [-1,1] 之外。

本探针只读，不改任何数据。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, param_to_norm, norm_to_param

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT = os.path.join(ROOT, "data", "finetune_v1")
V2 = os.path.join(ROOT, "data", "train_v2")

ft = TorchFAIDataset(FT, split="train")
print("finetune_v1 meta.config.dist_range =", ft.cfg["dist_range"])
print("finetune_v1 norm_consts dlo/dhi    =", ft.norm_consts["dlo"], ft.norm_consts["dhi"])

p_all = np.load(os.path.join(FT, "params.npy"), mmap_mode="r")
prac = np.asarray(p_all[:12000])
reg = np.asarray(p_all[12000:])
print("\n--- 真实 dist 分布 ---")
for name, arr in (("practice(0..11999)", prac), ("regular(12000..19999)", reg)):
    d = arr[:, 0]
    print(f"{name:22s} N={len(d):6d} min={d.min():.4f} max={d.max():.4f} "
          f"median={np.median(d):.4f}  >1.0m: {(d>1.0).mean()*100:5.1f}%")

# 用 finetune 自己的常数归一化，看有多少行越界
y = param_to_norm(p_all.astype(np.float64), ft.norm_consts)
out = (y[:, 0] > 1.0) | (y[:, 0] < -1.0)
print(f"\n--- 用 finetune_v1 自身常数归一化 ---")
print(f"dist 通道越界(|y|>1) 行数: {out.sum()} / {len(p_all)}"
      f"  ({out.mean()*100:.1f}%)")
print(f"  其中 practice 段: {out[:12000].sum()}, regular 段: {out[12000:].sum()}")
print(f"  y_dist 最大值 = {y[:,0].max():.4f}")

# 训练用 train_v2 的常数解码，看误差
v2 = TorchFAIDataset(V2, split="test")
print(f"\ntrain_v2 meta.config.dist_range = {v2.cfg['dist_range']}")
print(f"train_v2 norm_consts dlo/dhi     = {v2.norm_consts['dlo']:.4f}, {v2.norm_consts['dhi']:.4f}")

# 假设模型完美输出 finetune 归一化标签，用 train_v2 常数解码 → 系统误差多大
y_truth = param_to_norm(p_all.astype(np.float64), ft.norm_consts)
dec_v2 = norm_to_param(y_truth, v2.norm_consts)
err = np.abs(dec_v2[:, 0] - p_all[:, 0]) * 1000
print(f"\n--- 【关键】若网络完美预测 finetune 标签，用 train_v2 常数解码的 dist 误差 ---")
for name, sl in (("practice", slice(0, 12000)), ("regular", slice(12000, 20000))):
    e = err[sl]
    print(f"{name:10s} median={np.median(e):10.3f} mm  mean={e.mean():10.3f} mm  "
          f"max={e.max():10.3f} mm  >10mm: {(e>10).mean()*100:5.1f}%")

# train_v2 test 集本身用两套常数解码的差异
p_v2 = np.asarray(v2._mparams[v2._sel], dtype=np.float64)
y_v2_ft = param_to_norm(p_v2, ft.norm_consts)
d_v2 = norm_to_param(y_v2_ft, v2.norm_consts)[:, 0]
e2 = np.abs(d_v2 - p_v2[:, 0]) * 1000
print(f"\n--- train_v2 test(3900) 同一效应 ---")
print(f"median={np.median(e2):.3f} mm  mean={e2.mean():.3f} mm  max={e2.max():.1f} mm  "
      f">10mm: {(e2>10).mean()*100:.1f}%")
far = p_v2[:, 0] > 0.77
print(f"  仅远距(>0.77m, N={far.sum()}): median={np.median(e2[far]):.1f} mm  "
      f"mean={e2[far].mean():.1f} mm  max={e2[far].max():.1f} mm")
