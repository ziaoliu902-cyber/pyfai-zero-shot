# -*- coding: utf-8 -*-
"""v4w3 真实帧验收：7 帧 CeO₂ 粗解，v4w2 vs v4w3 同台对比。
判据（提前定死）：粗解偏差压到 <10mm → 精修可收敛。
v3far4（填缝前）: 150:-2.9 200:-25.9 250:-37.1 300:-49.5 350:-68.5 400:-45.4 450:-51.5
"""
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

from real_refine_v6 import load_px_nn, to_nn_input_wide
from torchfai.dataset import TorchFAIDataset as _T
from torchfai.dataset import norm_to_param

LAM = 0.972386                       # CeO₂ 波长 Å
OLD = {150: -2.9, 200: -25.9, 250: -37.1, 300: -49.5,
       350: -68.5, 400: -45.4, 450: -51.5}

consts = _T(os.path.join(ROOT, "data", "train_v4w3"), "val").norm_consts
m2, dev, _ = load_px_nn(r"runs\v4w2\best.pt"); m2.eval()
m3, _, _ = load_px_nn(r"runs\v4w3\best.pt"); m3.eval()

print(f"{'帧':>5s}{'GT_mm':>8s}{'v3far4':>9s}{'v4w2':>9s}{'v4w3':>9s}{'v3→w3':>8s}")
a2, a3, ao = [], [], []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt_mm = p.dist * 1000
    wide, k, _ = to_nn_input_wide(img)
    x = torch.from_numpy(np.log1p(wide[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu2, _ = m2(x, torch.tensor([LAM], device=dev), torch.tensor([k], device=dev))
        mu3, _ = m3(x, torch.tensor([LAM], device=dev), torch.tensor([k], device=dev))
    e2 = norm_to_param(mu2.double().cpu().numpy(), consts, k)[0, 0] * 1000 - gt_mm
    e3 = norm_to_param(mu3.double().cpu().numpy(), consts, k)[0, 0] * 1000 - gt_mm
    a2.append(abs(e2)); a3.append(abs(e3)); ao.append(abs(OLD[d]))
    print(f"{d:>5d}{gt_mm:8.1f}{OLD[d]:9.1f}{e2:+9.1f}{e3:+9.1f}{OLD[d]-e3:+8.1f}")

print(f"\n平均|Δdist|: v3far4={np.mean(ao):.1f}  v4w2={np.mean(a2):.1f}  "
      f"v4w3={np.mean(a3):.1f} mm")
ok = max(a3) < 10
print(f"判据（全帧 |Δ|<10mm）: {'✓ 通过' if ok else '✗ 未过'} "
      f"（v4w3 最差帧 {max(a3):.1f}mm）")
