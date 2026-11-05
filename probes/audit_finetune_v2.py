# -*- coding: utf-8 -*-
"""finetune_v2c / finetune_v2 / far_replay_v1 三卷审计（只读）。

七道关：
  A1 行数：各卷分片行数和 == meta.total == 打包 imgs.npy 行数
  A2 标签越界：并集常数下 |y| <= 1+1e-3 全成立（v1 事故防复发）
  A3 泄漏（行号级）：finetune_v2 far 段(20000..25999) 对 v3 val/test、v2 val/test 全为 0
  A4 重放语义：far 段行 ∈ base_v3 训练集 的占比（预期 ~70-80%，这是重放不是泄漏）
  A5 内容级抽验：far 段随机 32 行的 imgs/params 与 far_patch_v1 原行逐位一致
  A6 分布：各段相分布、dist 域、train split 远距(>0.77m)占比（对照 v1 的 6.2%）
  A7 切分互斥：三 split 两两交集 0（复算）
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, param_to_norm, _norm_consts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT2 = os.path.join(ROOT, "data", "finetune_v2")
FT2C = os.path.join(ROOT, "data", "finetune_v2c")
FAR_REPLAY = os.path.join(ROOT, "data", "far_replay_v1")
FARSRC = os.path.join(ROOT, "data", "far_patch_v1")

ok = [0, 0]


def check(name, cond, detail=""):
    ok[0 if cond else 1] += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def shard_total(root):
    import json
    tot = 0
    for f in sorted(os.listdir(root)):
        if f.startswith("shard_") and f.endswith(".npz"):
            with np.load(os.path.join(root, f)) as z:
                tot += int(len(z["params"]))
    meta = json.load(open(os.path.join(root, "meta.json"), encoding="utf-8"))
    imgs_n = int(np.load(os.path.join(root, "imgs.npy"), mmap_mode="r").shape[0]) \
        if os.path.exists(os.path.join(root, "imgs.npy")) else -1
    return tot, meta, imgs_n


print("== A1 行数 ==")
for root in (FT2C, FT2, FAR_REPLAY):
    t, meta, n = shard_total(root)
    if n == -1:  # 段目录（如 far_replay_v1）只有 npz 分片，无打包件，不要求 imgs.npy
        check(f"{os.path.basename(root)}: 分片和{t} == meta{meta['total']}（段目录，无打包件）",
              t == meta["total"])
    else:
        check(f"{os.path.basename(root)}: 分片和{t} == meta{meta['total']} == imgs{n}",
              t == meta["total"] == n)

print("== A2/A6 分布与标签（finetune_v2 并集常数） ==")
meta = __import__("json").load(open(os.path.join(FT2, "meta.json"), encoding="utf-8"))
cfg = meta["config"]
consts = _norm_consts(cfg)
params = np.load(os.path.join(FT2, "params.npy"))
y = param_to_norm(params.astype(np.float64), consts)
n_bad = int((np.abs(y) > 1 + 1e-3).sum())
check("A2 标签越界元素 = 0", n_bad == 0, f"|y|max={np.abs(y).max():.4f}")

dist = params[:, 0]
for name, sl in (("practice", slice(0, 12000)), ("regular", slice(12000, 20000)),
                 ("far_replay", slice(20000, 26000))):
    d = dist[sl]
    print(f"  段 {name:10s} n={sl.stop-sl.start:5d}  dist [{d.min():.2f},{d.max():.2f}]m  "
          f">0.77m: {(d>0.77).mean()*100:5.1f}%  中位 {np.median(d):.2f}m")

ds = TorchFAIDataset(FT2, split="train")
tr_rows = ds._sel
d_tr = dist[tr_rows]
far_share_train = (d_tr[tr_rows >= 20000] > 0.77).mean() if (tr_rows >= 20000).any() else 0
pct_far = (tr_rows >= 20000).mean() * 100
pct_dist_far = (d_tr > 0.77).mean() * 100
print(f"  finetune_v2 train({len(tr_rows)}): far 段占比 {pct_far:.1f}%，"
      f"dist>0.77m 占比 {pct_dist_far:.1f}%（v1 为 6.2%，目标 ~30%+）")

print("== A3/A4 泄漏与重放（行号级） ==")
v3_hold = np.unique(np.concatenate([
    TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split=sp)._sel
    for sp in ("val", "test")]))
v2_hold = np.unique(np.concatenate([
    TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split=sp)._sel
    for sp in ("val", "test")]))
v3_train = TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split="train")._sel
v2_train = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="train")._sel

# far_replay 行 k ← far_patch 行 sel[k] ← v2/v3 全局行 50000+sel[k]。
# 这里不重算 sel（merge 已固化），用内容级反查：far_replay 的 params 在 far_patch
# 中唯一定位（六元组撞车概率≈0），反推全局行号。
far_params_src = np.load(os.path.join(FARSRC, "params.npy")) \
    if os.path.exists(os.path.join(FARSRC, "params.npy")) else None
if far_params_src is None:  # far_patch_v1 无打包件 → 从分片拼
    fps = []
    for f in sorted(os.listdir(FARSRC)):
        if f.startswith("shard_"):
            with np.load(os.path.join(FARSRC, f)) as z:
                fps.append(z["params"])
    far_params_src = np.concatenate(fps)

fp = params[20000:26000].astype(np.float64)
src = far_params_src.astype(np.float64)
# 逐行定位（6000×15000 广播分块）
g_rows = np.empty(len(fp), dtype=np.int64)
for i in range(len(fp)):
    hit = np.where((np.abs(src - fp[i]) < 1e-9).all(axis=1))[0]
    assert len(hit) == 1, f"far 段第 {i} 行在 far_patch 中定位 {len(hit)} 次"
    g_rows[i] = hit[0]
global_rows = 50000 + g_rows
n_v3 = int(np.isin(global_rows, v3_hold).sum())
n_v2 = int(np.isin(global_rows, v2_hold).sum())
check("A3 far 段 ∩ train_v3 val/test = 0", n_v3 == 0, f"实际 {n_v3}")
check("A3 far 段 ∩ train_v2 val/test = 0", n_v2 == 0, f"实际 {n_v2}")
r3 = int(np.isin(global_rows, v3_train).sum())
r2 = int(np.isin(global_rows, v2_train).sum())
print(f"  A4 重放语义：far 段 ∈ base_v3 训练集 {r3}/{len(fp)} ({r3/len(fp)*100:.0f}%)，"
      f"∈ base_v2 训练集 {r2}/{len(fp)} ({r2/len(fp)*100:.0f}%)")

print("== A5 内容级抽验（far 段 32 行逐位） ==")
imgs2 = np.load(os.path.join(FT2, "imgs.npy"), mmap_mode="r")
# far_patch 分片偏移（用于按行定位原分片，避免全量拼内存）
fnames = sorted(f for f in os.listdir(FARSRC) if f.startswith("shard_"))
flens = []
for f in fnames:
    with np.load(os.path.join(FARSRC, f)) as z:
        flens.append(int(len(z["params"])))
foffs = np.concatenate([[0], np.cumsum(flens)])
rng = np.random.default_rng(7)
picks = rng.choice(len(fp), 32, replace=False)
diff, checked = 0, 0
by_shard = {}
for p in picks:
    si = int(np.searchsorted(foffs, g_rows[p], side="right") - 1)
    by_shard.setdefault(si, []).append(p)
for si, ps in by_shard.items():
    with np.load(os.path.join(FARSRC, fnames[si])) as z:
        for p in ps:
            row = int(g_rows[p] - foffs[si])
            diff += int(np.abs(imgs2[20000 + p].astype(np.int64) -
                               z["imgs"][row].astype(np.int64)).max())
            checked += 1
check("A5 far 段抽 32 行 imgs 逐位一致", diff == 0, f"行数 {checked}, max|Δ|={diff}")

print("== A7 切分互斥（复算） ==")
for root in (FT2C, FT2):
    sels = [TorchFAIDataset(root, split=sp)._sel for sp in ("train", "val", "test")]
    inter = sum(np.intersect1d(sels[i], sels[j]).size
                for i, j in ((0, 1), (0, 2), (1, 2)))
    check(f"{os.path.basename(root)} 三切分两两交集 = 0", inter == 0, f"实际 {inter}")

# practice/regular 与 v3 的参数级撞车检查（应为 0，两段独立生成）
# float32 字节级精确匹配（同一行必然逐位相同；独立生成撞车概率≈0）
v3keys = set(map(bytes, np.load(os.path.join(ROOT, "data", "train_v3", "params.npy"))))
hits = sum(1 for i in range(20000) if params[i].tobytes() in v3keys)
check("practice+regular 20k 行参数与 train_v3 撞车 = 0", hits == 0, f"实际 {hits}")

print(f"\n═══ 审计结果：{ok[0]} PASS / {ok[1]} FAIL ═══")
sys.exit(1 if ok[1] else 0)
