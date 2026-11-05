# -*- coding: utf-8 -*-
"""测量所有候选数据源的 dist 分布 / 相构成 / 实际工况画像率，为 finetune_v2 选回放源。

背景：v4b 用 finetune_v1 微调，评测时才发现两件事——
  (1) finetune_v1/meta.json 的 dist_range 写错（取 packs[0] 的 [0.1,1.0]，实际数据到 2.99m）
      → dist 标签编码与 v3 训练时不同，399 行标签越界(y>1)，这是真 bug；
  (2) practice 段 12k 张 dist 全 ≤1.0m（中位 0.136m），微调集 85% 集中在近距，
      远距档被带偏（v3 远距 MAE 17.64 → v4b 29.02）。

本脚本回答：远距回放该从哪拿、拿多少、会不会把相分布带偏。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset
from torchfai.phases import PHASE_NAMES

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stats(params, aux, name):
    d = params[:, 0]
    n = len(d)
    ph = np.bincount(aux[:, 0].astype(int), minlength=6)
    prac = (aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
    print(f"\n--- {name}  N={n} ---")
    print(f"  dist: min={d.min():.3f} 中位={np.median(d):.3f} max={d.max():.3f}"
          f"  >0.77m={np.mean(d>0.77)*100:.1f}%  >1.0m={np.mean(d>1.0)*100:.1f}%")
    print("  相: " + "  ".join(f"{PHASE_NAMES[i]}={ph[i]/n*100:.0f}%" for i in range(6)))
    print(f"  实际工况画像(6环+亮+无挡光): {prac.mean()*100:.1f}%")
    return dict(n=n, far=float(np.mean(d > 0.77)), prac=float(prac.mean()))


def from_mmap(root, split=None):
    ds = TorchFAIDataset(root, split=split or "train")
    sel = ds._sel
    if ds.mode == "mmap":
        return np.asarray(ds._mparams[sel]), np.asarray(ds._maux[sel])
    ps, ax = [], []
    for fn in ds.shards:
        with np.load(os.path.join(root, fn)) as z:
            ps.append(z["params"])
            ax.append(z["aux"])
    return np.concatenate(ps), np.concatenate(ax)


def from_npz(root):
    ps, ax = [], []
    for fn in sorted(f for f in os.listdir(root)
                     if f.startswith("shard_") and f.endswith(".npz")):
        with np.load(os.path.join(root, fn)) as z:
            ps.append(z["params"])
            ax.append(z["aux"])
    return np.concatenate(ps), np.concatenate(ax)


rows = []
for nm, root, kind in (
        ("train_v1 (v2/v3 基座 50k)", "data/train_v1", "mmap"),
        ("train_v2 (v3 基座 65k)", "data/train_v2", "mmap"),
        ("train_v3 (80k)", "data/train_v3", "mmap"),
        ("practice_patch/practice (12k 实际工况)", "data/practice_patch_v1/practice", "npz"),
        ("practice_patch/regular (8k 常规)", "data/practice_patch_v1/regular", "npz"),
        ("far_patch_v1 (15k 远距)", "data/far_patch_v1", "npz"),
        ("hard_patch/near_mix (12k 近距)", "data/hard_patch_v1/near_mix", "npz"),
        ("hard_patch/agbh_far (3k AgBh远)", "data/hard_patch_v1/agbh_far", "npz"),
):
    p, a = (from_mmap(os.path.join(ROOT, root)) if kind == "mmap"
            else from_npz(os.path.join(ROOT, root)))
    stats(p, a, nm)

# ---- 目标：微调集的 dist 结构应该贴近 v3 的训练结构，而不是贴近 practice 段 ----
print("\n" + "=" * 84)
print("目标配比设计（v3 训练集结构 vs 现状）")
print("=" * 84)
p3, a3 = from_mmap(os.path.join(ROOT, "data/train_v3"))
d3 = p3[:, 0]
print(f"  train_v3: >0.77m {np.mean(d3>0.77)*100:.1f}%   >1.0m {np.mean(d3>1.0)*100:.1f}%"
      f"   中位 {np.median(d3):.3f}m")
pf, af = from_npz(os.path.join(ROOT, "data/practice_patch_v1/practice"))
pr, ar = from_npz(os.path.join(ROOT, "data/practice_patch_v1/regular"))
dp, dr = pf[:, 0], pr[:, 0]
print(f"  practice: >0.77m {np.mean(dp>0.77)*100:.1f}%  >1.0m {np.mean(dp>1.0)*100:.1f}%"
      f"   中位 {np.median(dp):.3f}m")
print(f"  regular : >0.77m {np.mean(dr>0.77)*100:.1f}%  >1.0m {np.mean(dr>1.0)*100:.1f}%"
      f"   中位 {np.median(dr):.3f}m")
cur = np.concatenate([dp, dr])
print(f"\n  finetune_v1 现状(合): >0.77m {np.mean(cur>0.77)*100:.1f}%"
      f"  >1.0m {np.mean(cur>1.0)*100:.1f}%  中位 {np.median(cur):.3f}m")
print(f"  → 远距占比是 v3 的 {np.mean(cur>0.77)/np.mean(d3>0.77)*100:.0f}%"
      f"，这就是远距档退化的直接原因")

# ---- 模拟：从 far_patch_v1 抽 k 张回放，够不够把远距拉回 ----
pff, aff = from_npz(os.path.join(ROOT, "data/far_patch_v1"))
df = pff[:, 0]
print("\n  回放方案模拟（practice 12k + regular 8k + 回放 k 张远距）：")
for k in (3000, 5000, 8000, 12000):
    mix = np.concatenate([dp, dr, df[:k]])
    print(f"    回放 {k:5d} → 合集 N={len(mix):6d}  >0.77m {np.mean(mix>0.77)*100:4.1f}%"
          f"  >1.0m {np.mean(mix>1.0)*100:4.1f}%  (v3: 23.6% / 19.2%)")
