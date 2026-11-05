# -*- coding: utf-8 -*-
"""下限实验：初值好的样本 + 满迭代精修 → 6 参数最终停在哪一档。

逻辑：好初值避开假谷，引擎收敛后误差不再下降（迭代消融已证 400≈1200），
此时的残差水平 = 这套"图像+引擎"的噪声地板（光子统计极限）。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

ds = TorchFAIDataset("data/train_v2", split="test")
ck = torch.load("runs/base_v2/best.pt", map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

# 从全量成绩单挑"初值六参数都好"的样本
rows = [l.strip().split(",") for l in open("runs/refine_v2/results.csv",
       encoding="utf-8").readlines()[1:] if l.strip()]
good = []
for r in rows:
    eb = [float(x) for x in r[3:9]]
    if max(eb[0], eb[1], eb[2]) < 1.0 and max(eb[3], eb[4]) < 1.5:
        good.append(int(r[0]))
    if len(good) >= 6:
        break

refs = {}
res = []
print(f"挑出 {len(good)} 张初值好的样本（六参数初值误差均 <1mm / <1.5mrad），1500 迭代：\n")
for k in good:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    tp = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        mu, lv = model(torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda())
        sg = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
        pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, "data/train_v2")
    r = refs[ph].refine(img, pred, iters=1500, sigma_norm=sg, trust_k=0.5)
    e = np.abs(r["params"] - tp) * 1000
    res.append(e)
    print(f"#{k:<5d} {ph:8s} dist {e[0]:7.3f}mm | poni1 {e[1]:6.3f} poni2 {e[2]:6.3f}mm "
          f"| rot1 {e[3]:5.2f} rot2 {e[4]:5.2f} mrad")

res = np.array(res)
print("\n=== 下限（六样本的误差水平）===")
for i, nm in enumerate(("dist", "poni1", "poni2", "rot1", "rot2")):
    print(f"{nm:6s} 中位 {np.median(res[:, i]):8.3f}   最差 {res[:, i].max():8.3f}  "
          f"({'mm' if i < 3 else 'mrad'})")
