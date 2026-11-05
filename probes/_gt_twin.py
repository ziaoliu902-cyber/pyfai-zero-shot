# -*- coding: utf-8 -*-
"""GT 孪生终裁：synth_sample 钉在真实帧 GT 几何 → CNN 粗解。
孪生准 → 真实图像内容差(强度/峰宽/背景)；孪生也偏 → 数据管线约定 bug。"""
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

from real_refine_v6 import load_px_nn, match_gain
from torchfai.config import SimConfig
from torchfai.dataset import norm_to_param
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_detector, synth_sample

m, dev, consts = load_px_nn(r"runs\v3far4\last.pt")
m.eval()


def coarse_dist(canvas, k, lam_A):
    x = torch.from_numpy(
        np.log1p(canvas[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([lam_A], device=dev),
                  torch.tensor([k], device=dev))
    return float(norm_to_param(mu.double().cpu().numpy(), consts, k)[0, 0])


db = get_detector("Pilatus_1M", 2)
ph_i = PHASE_NAMES.index("CeO2")

for d in (250, 400):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    D = p.dist
    # GT poni 在 521×490 画布系（有效像素 834µm）
    PXEFF = 172e-6 / (521.0 / img.shape[0])
    from real_refine_v6 import to_nn_input_wide
    _, k, extra = to_nn_input_wide(img)
    fr1 = p.poni1 / (PXEFF * 521)
    fr2 = (p.poni2 - extra) / (PXEFF * 490)

    cfg = SimConfig()
    cfg.dist_range = (D * 0.999, D * 1.001)
    cfg.wavelength_range = (0.972386e-10, 0.972386e-10)
    cfg.poni_frac_range = (min(fr1, fr2) - 0.01, max(fr1, fr2) + 0.01)
    rng = np.random.default_rng(7)
    biases = []
    print(f"\n== ceria{d} GT={D*1000:.1f}mm k={k:.3f} "
          f"poni_frac=({fr1:.2f},{fr2:.2f}) ==")
    for t in range(4):
        s = synth_sample(rng, cfg, db, phase_idx=ph_i, pixel_scale=k)
        cd = coarse_dist(s["img"], k, 0.972386)
        biases.append((cd - D) * 1000)
        print(f"  孪生{t}: 粗解 {cd*1000:7.1f}mm "
              f"(偏 {(cd-D)*1000:+6.1f}mm) "
              f"σ={s['aux'][3] if len(s['aux'])>3 else '?'}")
    print(f"  孪生偏差中位 {np.median(biases):+.1f}mm   "
          f"真实帧实测偏 -37 ~ -45mm")
