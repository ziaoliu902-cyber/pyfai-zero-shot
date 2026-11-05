# -*- coding: utf-8 -*-
"""模糊假说 A/B：sim 角落样本 → 原生 vs 加「重采样往返模糊」→ 粗解偏移。
若加模糊后偏移复现 -30~-70mm，则病根=训练域缺重采样模糊，解=增广。"""
import os
import sys

import numpy as np
import torch
from scipy.ndimage import zoom

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
from real_refine_v6 import load_px_nn
from torchfai.dataset import TorchFAIDataset, norm_to_param

m, dev, consts = load_px_nn(r"runs\v3far4\last.pt")
m.eval()
ds = TorchFAIDataset("data/train_v3x", "val")


def coarse(img, lam, k):
    x = torch.from_numpy(img[None, None].astype(np.float32)).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([lam], device=dev),
                  torch.tensor([k], device=dev))
    return norm_to_param(mu.double().cpu().numpy(), consts, k)[0]


cand = []
for i in range(len(ds)):
    _, y, a = ds[i]
    y_ = y.numpy().astype(np.float64)
    a_ = a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    k = float(a_[6])
    d_true = float(norm_to_param(y_[None], consts, k)[0, 0])
    if 2.3 <= k <= 2.55 and 0.13 <= d_true <= 0.46:
        cand.append((i, k, d_true, float(a_[5])))
cand = cand[:25]
print(f"样本 {len(cand)} 个；模糊=6M往返(先×4.85上采再回521, order=1)")

d_native, d_blur = [], []
for i, k, d_true, lam in cand:
    img, _, _ = ds[i]
    img = img.numpy() if isinstance(img, torch.Tensor) else img
    lin = np.expm1(img.astype(np.float64))          # 回线性域再模糊
    f = 4.85                                        # ≈ 2527/521
    up = zoom(lin, (f, f), order=1)
    dn = zoom(up, (521 / up.shape[0], 490 / up.shape[1]), order=1)
    dn = np.log1p(np.clip(dn, 0, None))[:521, :490].astype(np.float32)
    p0 = coarse(img, lam, k)
    p1 = coarse(dn, lam, k)
    d_native.append((p0[0] - d_true) * 1000)
    d_blur.append((p1[0] - d_true) * 1000)
d_native, d_blur = np.array(d_native), np.array(d_blur)
print(f"原生:   均值{d_native.mean():+7.1f} 中位{np.median(d_native):+7.1f} mm")
print(f"加模糊: 均值{d_blur.mean():+7.1f} 中位{np.median(d_blur):+7.1f} mm")
print(f"偏移:   均值{(d_blur - d_native).mean():+7.1f} mm"
      f"（真实帧实测偏移 ≈ -30~-70）")
