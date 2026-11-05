# -*- coding: utf-8 -*-
"""build_sigq_true.py：组装 v4s2 精确标签 → data/train_v4w3/sigq.npy。

widenear4 重放真值 (20000,) 写入行区间 [79000, 99000)，底座 79000 行置 nan
（v4s2 训练时掩掉 σq 监督）。行偏移用 params 双向核对（pack 行 vs 分片 npz）。
旧 v1 测量标签备份为 sigq_meas_v1.npy。
"""
import json
import os
import shutil
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
os.chdir(ROOT)
DATA = os.path.join(ROOT, "data", "train_v4w3")
OFF = 79000

sig = np.load(os.path.join(ROOT, "probes", "results", "widenear4_sigq.npy"))
assert len(sig) == 20000, f"重放标签数 {len(sig)} ≠ 20000"

# ---- 行偏移双向核对 ----
params = np.load(os.path.join(DATA, "params.npy"))
with np.load(os.path.join(DATA, "shard_00311.npz")) as z:
    assert np.allclose(params[OFF], z["params"][0]), "行 79000 ≠ shard_00311 首行"
n389 = 0
with np.load(os.path.join(DATA, "shard_00389.npz")) as z:
    n389 = len(z["params"])
    assert np.allclose(params[OFF + 20000 - 1], z["params"][n389 - 1]), \
        "行 98999 ≠ shard_00389 末行"
print(f"偏移核对通过：79000↔shard_00311[0]，98999↔shard_00389[{n389-1}]")

arr = np.full(len(params), np.nan, np.float32)
arr[OFF:OFF + 20000] = sig

# 旧 v1 测量标签备份
v1 = os.path.join(DATA, "sigq.npy")
if os.path.isfile(v1):
    shutil.copy2(v1, os.path.join(DATA, "sigq_meas_v1.npy"))
np.save(v1, arr)
with open(os.path.join(DATA, "sigq_meta.json"), "w", encoding="utf-8") as f:
    json.dump(dict(lo=np.log10(0.002) - 0.1, hi=np.log10(0.06) + 0.1,
                   n_labeled=20000, offset=OFF, nan_rows=int(np.isnan(arr).sum()),
                   source="widenear4 replay (exact per-ring sigma, visible median)"),
              f, indent=1)
ok = arr[OFF:OFF + 20000]
print(f"sigq.npy 就位：有标签 {int(np.isfinite(arr).sum())} 行，"
      f"σq∈[{ok.min():.4f},{ok.max():.4f}] 中位 {np.median(ok):.4f}")
