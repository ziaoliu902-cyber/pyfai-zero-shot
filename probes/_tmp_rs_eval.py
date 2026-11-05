# -*- coding: utf-8 -*-
"""_tmp_rs_eval.py：径向对称闭式定心 val 全域评估——中心误差按离轴分箱。
对照：NN 环心误差（~20px 中位/58px 重尾）。"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
from solver_core import center_radial_symmetry
from torchfai.dataset import TorchFAIDataset

PX = 344e-6


def bc(d, p1, p2, r1, r2):
    v2 = -d * np.tan(r1)
    v1 = np.tan(r2) * d / np.cos(r1)
    return (p2 + v2) / PX - 0.5, (p1 + v1) / PX - 0.5


ds = TorchFAIDataset(r"D:\pyfai-project\data\train_v4w3", "val")
H, W = ds._mimgs.shape[1], ds._mimgs.shape[2]
sel = ds._sel
p_all = ds._mparams[sel]
cx, cy = bc(p_all[:, 0], p_all[:, 1], p_all[:, 2], p_all[:, 3], p_all[:, 4])
off = np.maximum(np.abs(cx - (W - 1) / 2) / W,
                 np.abs(cy - (H - 1) / 2) / H)
BINS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.45), (0.45, 2.5)]

N = 400
rng = np.random.default_rng(11)
sample = rng.choice(len(sel), size=min(N, len(sel)), replace=False)
errs = np.empty(len(sample))
t0 = time.time()
for n, i in enumerate(sample):
    img = np.asarray(ds._mimgs[int(sel[i])]).astype(np.float64)
    rcx, rcy = center_radial_symmetry(img)
    errs[n] = np.hypot(rcx - cx[i], rcy - cy[i])
print(f"RS 定心 {len(sample)} 样本，{time.time()-t0:.0f}s "
      f"({(time.time()-t0)/len(sample)*1000:.1f}ms/帧)")
print(f"整体: 中位={np.median(errs):.1f}px  p90={np.percentile(errs,90):.1f}px  "
      f"<10px 占比={(errs<10).mean()*100:.0f}%")
o = off[sample]
for b in BINS:
    m = (o >= b[0]) & (o < b[1])
    if m.sum() < 5:
        print(f"  离轴 {b[0]:.2f}~{b[1]:.2f}: n={m.sum()} (不足)")
        continue
    e = errs[m]
    print(f"  离轴 {b[0]:.2f}~{b[1]:.2f}: n={m.sum():3d}  "
          f"中位={np.median(e):6.1f}px  p90={np.percentile(e,90):7.1f}px  "
          f"<10px={(e<10).mean()*100:3.0f}%")
