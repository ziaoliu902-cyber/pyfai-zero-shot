# -*- coding: utf-8 -*-
"""λ 条件分布检查：宽域分层(k>1.5)的 λ 覆盖 vs 真实帧 0.9724Å。
若宽域样本 λ 集中在单值 → λ 条件外推 = 粗解偏差真凶。"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
aux = np.load(ROOT + r"\data\train_v3x\aux.npy", mmap_mode="r")
P = np.load(ROOT + r"\data\train_v3x\params.npy", mmap_mode="r")
k = np.asarray(aux[:, 6], dtype=float)
lam = np.asarray(aux[:, 5], dtype=float) * 1e10      # Å
dist = np.asarray(P[:, 0], dtype=float)

wide = k > 1.5
print(f"宽域样本 {wide.sum()} 个的 λ 分布(Å):")
h, e = np.histogram(lam[wide], bins=10)
for i in range(10):
    bar = "#" * int(50 * h[i] / max(h.max(), 1))
    print(f"  {e[i]:5.2f}~{e[i+1]:5.2f} {h[i]:6d} {bar}")
print(f"  唯一值数: {len(np.unique(np.round(lam[wide], 4)))}")
print(f"\n窄域(k<=1.5) λ 分布: min={lam[~wide].min():.3f} "
      f"max={lam[~wide].max():.3f} "
      f"唯一值数={len(np.unique(np.round(lam[~wide], 4)))}")
# 真实 ceria 宽帧: λ=0.9724, k=2.425, d=0.15~0.45
sub = wide & (np.abs(lam - 0.9724) < 0.15)
print(f"\n宽域 & λ∈[0.82,1.12](真实帧区): {sub.sum()} 个")
sub2 = wide & (dist < 0.5) & (np.abs(lam - 0.9724) < 0.15)
print(f"再加 dist<0.5m: {sub2.sum()} 个")
