# -*- coding: utf-8 -*-
"""v4w 真实帧验收：7 帧 CeO₂ 粗解 vs v3far4 崩溃记录。
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
from torchfai.dataset import norm_to_param

m, dev, consts = load_px_nn(r"runs\v4w2\best.pt")
consts = None
from torchfai.dataset import TorchFAIDataset as _T
consts = _T(os.path.join(ROOT, "data", "train_v4w2"), "val").norm_consts
m.eval()
LAM = 0.972386                       # CeO₂ 波长 Å
OLD = {150: -2.9, 200: -25.9, 250: -37.1, 300: -49.5,
       350: -68.5, 400: -45.4, 450: -51.5}

print(f"{'帧':>9s}{'k':>6s}{'GT_mm':>8s}{'v3far4':>9s}{'v4w':>9s}{'改善':>8s}"
      f"{'Δponi1_px':>10s}{'Δponi2_px':>10s}{'Δrot3_mrad':>11s}")
tot_new, tot_old = [], []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt_mm = p.dist * 1000
    wide, k, extra_col_m = to_nn_input_wide(img)
    x = torch.from_numpy(np.log1p(wide[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([LAM], device=dev),
                  torch.tensor([k], device=dev))
    q = norm_to_param(mu.double().cpu().numpy(), consts, k)[0]
    err = (q[0] * 1000 - gt_mm)
    tot_new.append(abs(err))
    tot_old.append(abs(OLD[d]))
    # poni 换算到画布有效像素系（行不裁；列左裁 extra_col_m）
    PX6M = 0.000172                    # 6M 原生像素（与 v6 脚本一致）
    eff_px = k * consts["Hm"] * 490 / wide.shape[1] if False else None
    # 直接用 k 推 eff_px：to_nn_input_wide 里 eff_px = k*Hm0/ (521/H)*(490/W')
    # 简化：eff_px = PX6M / f，f=521/H → 由 k 定义反推
    f = 521.0 / img.shape[0]
    eff_px = PX6M / f
    dp1 = (q[1] - p.poni1) / eff_px
    dp2 = (q[2] - (p.poni2 - extra_col_m)) / eff_px
    print(f"{d:>9d}{k:6.3f}{gt_mm:8.1f}{OLD[d]:9.1f}{err:+9.1f}"
          f"{OLD[d]-err:+8.1f}{dp1:10.1f}{dp2:10.1f}{q[5]*1000:11.2f}")
print(f"\n平均|Δdist|: v3far4={np.mean(tot_old):.1f}mm → v4w={np.mean(tot_new):.1f}mm"
      f" （{np.mean(tot_old)/max(np.mean(tot_new),0.01):.1f}×）")
