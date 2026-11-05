# -*- coding: utf-8 -*-
"""缝隙实验：真实宽帧的模块暗缝检测+填充 → 粗解偏差是否消失。
缝 guilty：填缝后偏 -45 → ~0；innocent：不变。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI

from real_refine_v6 import load_px_nn, match_gain, to_nn_input_wide
from torchfai.dataset import norm_to_param

m, dev, consts = load_px_nn(r"runs\v3far4\last.pt")
m.eval()


def coarse_dist(canvas, k):
    x = torch.from_numpy(
        np.log1p(canvas[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([0.972386], device=dev),
                  torch.tensor([k], device=dev))
    return float(norm_to_param(mu.double().cpu().numpy(), consts, k)[0, 0])


def fill_dark_bands(a, frac=0.25):
    """行/列方向中位强度低于全局中位 frac 的暗带 → 邻均值填充。"""
    a = a.astype(np.float64).copy()
    gmed = np.median(a)
    bad_r = np.where(np.median(a, axis=1) < frac * gmed)[0]
    bad_c = np.where(np.median(a, axis=0) < frac * gmed)[0]
    for r in bad_r:                      # 行填充：上下邻行均值
        up = a[r - 1] if r > 0 else a[r + 1]
        dn = a[r + 1] if r < a.shape[0] - 1 else a[r - 1]
        a[r] = 0.5 * (up + dn)
    for c in bad_c:
        lf = a[:, c - 1] if c > 0 else a[:, c + 1]
        rt = a[:, c + 1] if c < a.shape[1] - 1 else a[:, c - 1]
        a[:, c] = 0.5 * (lf + rt)
    return a, len(bad_r), len(bad_c)


for d in (250, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt_mm = p.dist * 1000
    wide, k, _ = to_nn_input_wide(img)
    wide = match_gain(wide)
    fixed, nr, nc = fill_dark_bands(wide)
    b0 = (coarse_dist(wide, k) * 1000 - gt_mm)
    b1 = (coarse_dist(fixed, k) * 1000 - gt_mm)
    print(f"ceria{d}: 填缝(行{nr}/列{nc}) 偏差 {b0:+.1f} → {b1:+.1f}mm")
