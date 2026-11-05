# -*- coding: utf-8 -*-
"""评估：NN 几何回归在 ceria 七档上的 ΔD，和经典路径对比；看融合有没有收益。"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
import pyFAI
import torch

from real_refine_v6 import load_px_nn, to_nn_input_wide
from torchfai.dataset import TorchFAIDataset, norm_to_param

LAM = 0.972386
MODEL = os.path.join(ROOT, "runs", "v4s1", "best.pt")

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v4w3"), "val")
m, dev, _ = load_px_nn(MODEL)
m.eval()

rows = []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    wide, k, _ = to_nn_input_wide(img)
    x = torch.from_numpy(np.log1p(wide[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, lv = m(x, torch.tensor([LAM], device=dev),
                   torch.tensor([k], device=dev))
    p6 = norm_to_param(mu.double().cpu().numpy(), ds.norm_consts, k)[0]
    sig = np.exp(0.5 * lv.float().cpu().numpy())[0]
    D_nn = p6[0] * 1000
    cx_nn, cy_nn = p6[2] * 1e6 / 172.0, p6[1] * 1e6 / 172.0

    # 经典解（phase_fit 跑出来的 poni）
    cp = pyFAI.load(os.path.join(ROOT, "probes", "results", f"ceria_{d}.poni"))
    D_cl = cp.dist * 1000
    cx_cl, cy_cl = cp.poni2 / 172e-6, cp.poni1 / 172e-6

    rows.append((d, gt, D_cl, D_nn, sig[0], cx_cl, cy_cl, cx_nn, cy_nn))
    print(f"{d:>4} GT={gt:7.2f}  经典 ΔD={D_cl-gt:+7.2f}  "
          f"NN ΔD={D_nn-gt:+8.2f}  NN σ_lnD={sig[0]:.3f}  "
          f"NN中心=({cx_nn:.0f},{cy_nn:.0f}) 经典中心=({cx_cl:.0f},{cy_cl:.0f})")

a = np.array([abs(r[2] - r[1]) for r in rows])
b = np.array([abs(r[3] - r[1]) for r in rows])
print(f"\n经典 |ΔD| 平均 {a.mean():.2f} 最差 {a.max():.2f}")
print(f"NN   |ΔD| 平均 {b.mean():.2f} 最差 {b.max():.2f}")

# 简单融合：以 NN 的预测为中心、经典为基准，看线性组合 α·经典+(1-α)·NN
print("\n融合扫描（α 为经典权重）:")
for al in np.linspace(0, 1, 11):
    e = [abs((al * r[2] + (1 - al) * r[3]) - r[1]) for r in rows]
    print(f"  α={al:.1f}  平均|ΔD|={np.mean(e):5.2f}  最差={np.max(e):6.2f}")
