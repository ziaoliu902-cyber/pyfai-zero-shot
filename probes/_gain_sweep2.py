# -*- coding: utf-8 -*-
"""增益扫描终审：真实画布整体缩放 s → 粗解 dist 怎么动？
若存在一致的 s* 使 7 帧偏差同归零 → 亮度标定=真凶（治=改 match_gain 策略）。
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
from torchfai.dataset import TorchFAIDataset as _T
consts = _T(os.path.join(ROOT, "data", "train_v4w2"), "val").norm_consts
m.eval()
LAM = 0.972386

# 训练域亮度锚：sim 卷 log1p 中位（dataset 同款处理）
tr = _T(os.path.join(ROOT, "data", "train_v4w2"), "val")
meds = []
for i in range(0, len(tr), 97):
    img, _, _ = tr[i]
    im = img.numpy() if hasattr(img, "numpy") else img
    meds.append(np.median(np.log1p(im)))
print(f"训练域 log1p 中位: {np.median(meds):.2f} (n={len(meds)})")

SWEEPS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
frames = {}
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    wide, k, extra = to_nn_input_wide(img)          # 已含 p99.9 match_gain
    frames[d] = (wide, k, p.dist * 1000)

hdr = "帧\\s " + "".join(f"{s:>8.2f}" for s in SWEEPS)
print(hdr)
s_star = {}
for d, (wide, k, gt) in frames.items():
    row = f"{d:>5d} "
    best = (1e9, None)
    for s in SWEEPS:
        x = torch.from_numpy(
            np.log1p(wide * s)[None, None].astype(np.float32)).to(dev)
        with torch.no_grad():
            mu, _ = m(x, torch.tensor([LAM], device=dev),
                      torch.tensor([k], device=dev))
        pred = float(norm_to_param(mu.double().cpu().numpy(), consts, k)[0, 0])
        err = pred * 1000 - gt
        row += f"{err:>+8.1f}"
        if abs(err) < abs(best[0]):
            best = (err, s)
    s_star[d] = best
    print(row)
print("\n帧级最优 s*（偏差最小点）:")
for d, (e, s) in s_star.items():
    print(f"  {d}mm: s*={s} (残差{e:+.1f}mm)")
