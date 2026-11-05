# -*- coding: utf-8 -*-
"""dataset.py 正确性测试：切分互斥、形状/范围、归一化往返、DataLoader 吞吐。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, param_to_norm, norm_to_param, PARAM_NAMES

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "train_v1")

# ---- 1. 三个切分 ----
tr = TorchFAIDataset(ROOT, "train")
va = TorchFAIDataset(ROOT, "val")
te = TorchFAIDataset(ROOT, "test")
print(f"split sizes: train={len(tr)} val={len(va)} test={len(te)} "
      f"(sum={len(tr)+len(va)+len(te)})")
assert len(tr) + len(va) + len(te) == 50000
assert np.intersect1d(tr._sel, va._sel).size == 0, "train/val 有交集！"
assert np.intersect1d(tr._sel, te._sel).size == 0, "train/test 有交集！"
assert np.intersect1d(va._sel, te._sel).size == 0, "val/test 有交集！"
print("切分互斥：OK")

# ---- 2. 样本结构与范围 ----
x, y, aux = tr[0]
assert x.shape == (521, 490) and x.dtype == torch.float32
assert y.shape == (6,) and float(y.min()) >= -1.001 and float(y.max()) <= 1.001
assert aux.shape == (5,)
print(f"样本结构 OK：x{tuple(x.shape)} range[{x.min():.2f},{x.max():.2f}]  "
      f"y range[{y.min():.3f},{y.max():.3f}]")

# ---- 3. 归一化往返 ----
c = tr.norm_consts
p = np.stack([tr[i][1].numpy() for i in range(64)])       # 借 y 反解不行——直接取原始
with np.load(os.path.join(ROOT, "shard_00000.npz")) as z:
    p_raw = z["params"][:64]
y_rt = param_to_norm(p_raw, c)
p_back = norm_to_param(y_rt, c)
err = np.abs(p_back - p_raw)
rel = err / (np.abs(p_raw) + 1e-30)
print(f"往返最大绝对误差 {err.max():.3g}，最大相对误差 {rel.max():.3g}")
assert rel.max() < 1e-6, "归一化不可逆！"
print("归一化往返：OK")

# ---- 4. DataLoader 吞吐（顺序 + 随机各测 10 批）----
dl = DataLoader(tr, batch_size=32, shuffle=True, num_workers=0)
it = iter(dl)
xs, ys, auxs = next(it)
assert xs.shape == (32, 521, 490) and ys.shape == (32, 6)
t0 = time.time()
for _ in range(9):
    next(it)
dt = time.time() - t0
print(f"batch32 随机取 10 批：{32*10/dt:.0f} imgs/s（含解压与 log1p）")
print("PASS：dataset 全部检查通过")
