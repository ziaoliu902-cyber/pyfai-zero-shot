# -*- coding: utf-8 -*-
"""px_v2 四诊：AgBh 增益扫描 + alumina 画布布局假设（贴边框 vs 拉满改 k）。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabio

from real_refine_v6 import REAL, ROOT, load_px_nn, match_gain
from real_refine_v5 import to_nn_input_phys
from torchfai.dataset import norm_to_param

m, dev, consts = load_px_nn(os.path.join(ROOT, "runs/px_v2/best.pt"))


def pred(img, lam, k):
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].to(dev)
        mu, _ = m(x, torch.tensor([lam], device=dev),
                  torch.tensor([k], device=dev))
    return norm_to_param(mu.float().cpu().numpy(), consts, k)[0]


print("== AgBh：输入体检 + 增益扫描 ==")
img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
raw = to_nn_input_phys(img)
print(f"   raw: max={raw.max():.0f} mean={raw.astype(np.float64).mean():.2f} "
      f"p99.9={np.percentile(raw, 99.9):.0f} zerofrac={(raw==0).mean():.3f}")
g = match_gain(raw)
base = g / max(np.percentile(raw, 99.9) / 20536.0, 1e-6)   # 原始
for scale in (0.5, 1.0, 2.0, 4.0):
    p = pred(g * scale / scale * scale if False else g * scale, 1.0, 1.0)
    print(f"   gain×{scale}: pred={p[0]*1000:7.1f}mm d={(p[0]-1.583)*1000:+7.1f}mm")

print("== alumina：布局假设 ==")
AL = os.path.join(REAL, "alumina")
dark = fabio.open(os.path.join(AL, "dark_0001.edf")).data.astype(np.float64)
frames = [fabio.open(os.path.join(AL, f"al2o3_{i:04d}.edf")).data
          for i in range(51)]
aimg = np.clip(np.mean(frames, 0) - dark, 0, None).astype(np.float32)
del frames
p1, p2 = 103.358e-6, 102.53e-6
cy, cx = 0.0529565255372 / p1, 0.05473342483 / p2
H, W = aimg.shape
yy0, xx0 = np.mgrid[0:H, 0:W]
r_m = np.sqrt(((yy0 - cy) * p1) ** 2 + ((xx0 - cx) * p2) ** 2)
aimg[r_m < 0.006] = 0.0
print(f"   Frelon 帧: {H}x{W}")
from scipy.ndimage import zoom as szoom

# 布局1（现行）：物理映射 k=1，零边框
h_t, w_t = int(round(H * p1 / 344e-6)), int(round(W * p2 / 344e-6))
xs = szoom(aimg, (h_t / H, w_t / W), order=1)
c1 = np.zeros((521, 490), np.float32)
r0, c0 = (521 - h_t) // 2, (490 - w_t) // 2
c1[r0:r0 + h_t, c0:c0 + w_t] = xs
p = pred(match_gain(c1), 0.70848, 1.0)
print(f"   L1 物理贴放(k=1,边框{521-h_t}px): {p[0]*1000:7.1f}mm "
      f"d={(p[0]-0.1169)*1000:+7.1f}mm")

# 布局2：拉满画布，k 按有效像素折算（外推 k<1）
f2 = 521 / H
xs2 = szoom(aimg, (f2, f2), order=1)[:, :490]
eff = p1 / f2
k2 = eff / 344e-6
p = pred(match_gain(xs2), 0.70848, k2)
print(f"   L2 拉满画布(k={k2:.2f},无边框): {p[0]*1000:7.1f}mm "
      f"d={(p[0]-0.1169)*1000:+7.1f}mm")
