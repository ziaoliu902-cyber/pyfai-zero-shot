# -*- coding: utf-8 -*-
"""v4s2 真实验收：ceria 八档粗解 v4w3 vs v4s2 + σq 头读数。
判据（预注册）：平均|Δ|<10mm 且最差帧<15mm。
附带：out_sig（归一化 log10 环宽预测）→ 真实帧环宽是否在训练域内。
"""
import json
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

LAM = 0.972386
ds = _T(os.path.join(ROOT, "data", "train_v4w3"), "val")
consts = ds.norm_consts
with open(os.path.join(ROOT, "data", "train_v4w3", "sigq_meta.json"),
          encoding="utf-8") as f:
    sm = json.load(f)
m3, dev, _ = load_px_nn(r"runs\v4w3\best.pt"); m3.eval()
m1, _, _ = load_px_nn(r"runs\v4s2\best.pt"); m1.eval()


def denorm_sig(v):
    return 10 ** ((v + 1) / 2 * (sm["hi"] - sm["lo"]) + sm["lo"])


print(f"{'帧':>5}{'GT_mm':>8}{'v4w3':>9}{'v4s2':>9}{'w3→s1':>8}"
      f"{'σq真值预测':>9}{'域内':>4}")
a3, a1 = [], []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    wide, k, _ = to_nn_input_wide(img)
    x = torch.from_numpy(np.log1p(wide[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu3, _ = m3(x, torch.tensor([LAM], device=dev), torch.tensor([k], device=dev))
        mu1, _ = m1(x, torch.tensor([LAM], device=dev), torch.tensor([k], device=dev))
        w_hat = denorm_sig(float(m1.out_sig.view(-1)[0]))
    e3 = norm_to_param(mu3.double().cpu().numpy(), consts, k)[0, 0] * 1000 - gt
    e1 = norm_to_param(mu1.double().cpu().numpy(), consts, k)[0, 0] * 1000 - gt
    a3.append(abs(e3)); a1.append(abs(e1))
    in_dom = "✓" if -1 <= float(m1.out_sig.view(-1)[0]) <= 1 else "✗"
    print(f"{d:>5}{gt:8.1f}{e3:+9.1f}{e1:+9.1f}{e3-e1:+8.1f}{w_hat:9.4f}{in_dom:>4}")

print(f"\n平均|Δ|: v4w3={np.mean(a3):.1f} → v4s2={np.mean(a1):.1f} mm"
      f" （最差 {max(a1):.1f}）")
ok = np.mean(a1) < 10 and max(a1) < 15
print(f"判据（均值<10 且最差<15）: {'✓ 通过' if ok else '✗ 未过'}")
