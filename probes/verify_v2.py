# -*- coding: utf-8 -*-
"""train_v2 独立验收：far 样本逐位对拍 + 分布检查。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "data", "train_v2")
FAR = os.path.join(ROOT, "data", "far_patch_v1")

ok = True

# ---- 1) far 样本逐位对拍：dataset 取样 vs 原始分片 ----
ds = TorchFAIDataset(V2, split="test")
far_sel_local = np.where(ds._sel >= 50000)[0]
i0 = int(far_sel_local[3])          # 任取一张 far 考题
g = int(ds._sel[i0])                # 全局行号
row_in_far = g - 50000
x, y, aux = ds[i0]                  # x: log1p 后的 float32

# 从 far 分片直接找到同一样本
shards = sorted(f for f in os.listdir(FAR) if f.startswith("shard_"))
acc = 0
for fn in shards:
    with np.load(os.path.join(FAR, fn)) as z:
        n = len(z["params"])
        if acc + n > row_in_far:
            img_raw = z["imgs"][row_in_far - acc]
            p_raw = z["params"][row_in_far - acc]
            a_raw = z["aux"][row_in_far - acc]
            break
        acc += n

img_direct = np.log1p(img_raw.astype(np.float32))
same_img = np.array_equal(img_direct, x.numpy())
same_aux = np.array_equal(a_raw, aux.numpy())
print(f"[对拍] 全局行号 {g}（far 第 {row_in_far} 张）  图像逐位一致={same_img}  aux一致={same_aux}")
ok &= same_img and same_aux

# 归一化往返：dataset 的 y 反变换应还原 p_raw
p_back = __import__("torchfai.dataset", fromlist=["norm_to_param"]).norm_to_param(
    y.numpy().astype(np.float64), ds.norm_consts)
err = np.abs(p_back - p_raw.astype(np.float64)) / np.abs(p_raw)
print(f"[对拍] 6参数归一化往返最大相对误差 {err.max():.2e}  ({'PASS' if err.max() < 1e-4 else 'FAIL'})")
ok &= err.max() < 1e-4

# ---- 2) 分布检查 ----
params = np.load(os.path.join(V2, "params.npy"), mmap_mode="r")
aux = np.load(os.path.join(V2, "aux.npy"))
d_far = np.asarray(params[50000:, 0])
d_v1 = np.asarray(params[:50000, 0])
print(f"[分布] far 距离: min={d_far.min():.2f} max={d_far.max():.2f} 均值={d_far.mean():.2f} (应≈0.6-3.0)")
print(f"[分布] v1  距离: min={d_v1.min():.2f} max={d_v1.max():.2f} (应≈0.1-3.0)")
far_phase_ok = bool((aux[50000:, 0] == 5).all())
print(f"[分布] far 全部为 AgBh(idx5): {far_phase_ok}")

# 各 split 的 far 距离分布（确认考试卷远距题充足且覆盖全程）
for sp in ("val", "test"):
    s = TorchFAIDataset(V2, split=sp)
    gsel = s._sel[s._sel >= 50000] - 50000
    dd = d_far[gsel]
    print(f"[分布] {sp}: far N={len(dd)}  距离 {dd.min():.2f}-{dd.max():.2f}m  >1.5m 占比 {float((dd>1.5).mean())*100:.0f}%")

# ---- 3) 磁盘占用 ----
total = sum(os.path.getsize(os.path.join(V2, f)) for f in os.listdir(V2))
print(f"[磁盘] train_v2 名义大小 {total/1e9:.1f}GB（硬链接部分不额外占盘）")

print("\n验收结论:", "ALL PASS" if ok else "FAIL")
