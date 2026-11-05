# -*- coding: utf-8 -*-
"""拆账演示：总值(nat) = Σ 每参数 0.5[(err/σ)²+logσ²]，与面板 MAE(物理) 的关系。
取 val 卷 3 个样本，读 v3far4 best.pt 的 mu/logvar 头逐项打印。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
from real_refine_v6 import load_px_nn
from torchfai.dataset import TorchFAIDataset, norm_to_param

NAMES = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")

m, dev, consts = load_px_nn(r"runs\v3far4\best.pt")
m.eval()
ds = TorchFAIDataset("data/train_v3x2", "val")

tot_terms = np.zeros(6)
shown = False
errs_p, sigs_p = [], []
for i in (100, 1000, 2000, 3000):
    img, y, a = ds[i]
    img = img.numpy() if isinstance(img, torch.Tensor) else img
    y = y.numpy() if isinstance(y, torch.Tensor) else y
    a = a.numpy() if isinstance(a, torch.Tensor) else a
    # 数据集已做 log1p（训练循环直接 unsqueeze 喂模）——不能再变换
    x = torch.from_numpy(img[None, None].astype(np.float32)).to(dev)
    with torch.no_grad():
        mu, logvar = m(x,
                       torch.tensor([a[5]], device=dev, dtype=torch.float32),
                       torch.tensor([a[6]], device=dev, dtype=torch.float32))
    mu, logvar = mu[0].cpu().numpy(), logvar[0].cpu().numpy()
    sig = np.exp(logvar / 2)                       # σ，归一化域
    err = np.abs(mu - y)                           # 误差，归一化域
    term = 0.5 * (logvar + (err / sig) ** 2)       # 每参数 NLL 项
    tot_terms += term
    # 物理域换算（用同一 consts/k 反归一化，差分即物理尺度）
    k = float(a[6])
    sc = (norm_to_param(np.eye(6) * 0 + np.array([0.5, 0, 0, 0, 0, 0]), consts, k)
          - norm_to_param(np.zeros((1, 6)), consts, k))[0, 0] * 2  # 粗略 dist 半程
    errs_p.append(err)
    sigs_p.append(sig)
    if not shown:
        print(f"{'参数':6s}{'σ(归一)':>9s}{'|误差|(归一)':>12s}"
              f"{'(err/σ)²':>10s}{'logσ²':>9s}{'NLL项':>8s}")
        for j, nm in enumerate(NAMES):
            print(f"{nm:6s}{sig[j]:>9.3f}{err[j]:>12.3f}"
                  f"{(err[j]/sig[j])**2:>10.2f}{logvar[j]:>9.2f}{term[j]:>8.2f}")
        shown = True

print(f"\n样本均值 Σ NLL项 = {tot_terms.mean():.2f} nat  ← 总值（val 域实测 -9.8）")
print(f"σ/误差 中位比 = {np.median(np.array(sigs_p)/np.maximum(np.array(errs_p),1e-9)):.2f}"
      f"  （≈1.0 = 校准良好：σ 就是它自己误差的预测器）")
