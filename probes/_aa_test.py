# -*- coding: utf-8 -*-
"""抗锯齿终审：to_nn_input_wide 的 scipy zoom(order=1) 4.85× 下采样无抗锯齿
→ 高q密环混叠。换 PIL 双线性（真面积平均）重采样，看 7 帧偏差是否塌缩。
若塌缩 → 皮=管线混叠，模型/训练全不用动，改预处理即治愈。
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
from PIL import Image

from real_refine_v6 import load_px_nn, match_gain, to_nn_input_wide
from torchfai.dataset import norm_to_param

m, dev, consts = load_px_nn(r"runs\v4w2\best.pt")
from torchfai.dataset import TorchFAIDataset as _T
consts = _T(os.path.join(ROOT, "data", "train_v4w2"), "val").norm_consts
m.eval()
LAM = 0.972386
PX6M = 0.000172


def canvas_aa(img, hw=(521, 490), flt=Image.LANCZOS):
    """抗锯齿版：PIL 重采样 → 中央裁 → match_gain。"""
    H, W = img.shape
    f = hw[0] / H
    W2 = int(round(W * f))
    im = Image.fromarray(np.clip(img, 0, None).astype(np.float64))
    b = np.asarray(im.resize((W2, hw[0]), flt), dtype=np.float64)
    c0 = (W2 - hw[1]) // 2
    return match_gain(b[:, c0:c0 + hw[1]]).astype(np.float32)


print(f"{'帧':>5s}{'旧zoomΔmm':>10s}{'抗锯齿Δmm':>11s}{'改善':>8s}")
tot_o, tot_n = [], []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    wide, k, extra = to_nn_input_wide(img)
    aa = canvas_aa(img)
    outs = []
    for canvas in (wide, np.clip(aa, 0, None)):
        x = torch.from_numpy(
            np.log1p(canvas[None, None].astype(np.float32))).to(dev)
        with torch.no_grad():
            mu, _ = m(x, torch.tensor([LAM], device=dev),
                      torch.tensor([k], device=dev))
        outs.append(float(norm_to_param(
            mu.double().cpu().numpy(), consts, k)[0, 0]) * 1000 - gt)
    o, n = outs
    tot_o.append(abs(o)); tot_n.append(abs(n))
    print(f"{d:>5d}{o:+10.1f}{n:+11.1f}{o-n:+8.1f}")
print(f"\n平均|Δ|: 旧zoom={np.mean(tot_o):.1f} → 抗锯齿={np.mean(tot_n):.1f}mm")
