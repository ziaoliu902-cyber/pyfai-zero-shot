# -*- coding: utf-8 -*-
"""增益扫描：真实宽帧 match_gain 后再乘不同系数 → 粗解 dist 偏移。
若强敏感 → 病根=增益标定；解=亮度不变性训练或更好的归一化。"""
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

from real_refine_v6 import (load_px_nn, match_gain, to_nn_input_wide)

m, dev, consts = load_px_nn(r"runs\v3far4\last.pt")
m.eval()


def coarse_dist(img_canvas, k):
    x = torch.from_numpy(
        np.log1p(img_canvas[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([0.972386], device=dev),
                  torch.tensor([k], device=dev))
    from torchfai.dataset import norm_to_param
    return float(norm_to_param(mu.double().cpu().numpy(), consts, k)[0, 0])


for d in (250, 400):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt_mm = p.dist * 1000
    wide, k, _ = to_nn_input_wide(img)
    wide = match_gain(wide)                    # 现行管线基准
    print(f"\n== ceria {d}mm (GT {gt_mm:.1f}mm, k={k:.3f}) ==")
    for f in (0.25, 0.5, 1.0, 2.0, 4.0):
        cd = coarse_dist(wide * f, k)
        print(f"  增益×{f:4.2f} → 粗解 {cd*1000:7.1f}mm "
              f"(偏 {(cd*1000-gt_mm):+6.1f}mm)")
