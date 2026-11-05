# -*- coding: utf-8 -*-
"""实际工况画像子集实测：多环 + 高计数 + 无驼峰（真实标定的样子）。

问题：信息底（poni ~1px / dist 漂移 mm 级）是"全体的底"还是
     "困难尾部的底"？清晰多环样本上精修器是否本来就到 manual 水平？
对照：
  P1 实际工况子集上 C0（NN+0.5σ笼）
  P2 实际工况子集上 GT 起点（无笼）漂移
  参考：全体中位（全量 results.csv：poni 1.16px；GT 漂移 2.53px）
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NMAX = 30

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
aux = ds._maux[ds._sel]
nvis, imax, hump = aux[:, 1], aux[:, 3], aux[:, 4]

prac = (nvis >= 6) & (imax >= 10000) & (hump < 0)
print(f"测试集 N={len(aux)}  实际工况画像(多环≥6 & 亮≥10k & 无驼峰): "
      f"{prac.sum()} 张 ({prac.mean()*100:.1f}%)")
print(f"全集中 n_vis 分布: " +
      " ".join(f"P{p}={np.percentile(nvis,p):.0f}" for p in (10, 50, 90)) +
      f"   imax: " +
      " ".join(f"P{p}={np.percentile(imax,p)/1000:.0f}k" for p in (10, 50, 90)) +
      f"   驼峰占比 {(hump>=0).mean()*100:.0f}%")

idx = np.where(prac)[0]
rng = np.random.default_rng(11)
sel = sorted(rng.choice(idx, min(NMAX, len(idx)), replace=False).tolist())

ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

data, refs = [], {}
for k in sel:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    true_p = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
        mu, lv = model(x)
        pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64),
                             ds.norm_consts)
        sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
    data.append((img, true_p, ph, pred, sig))

PX_UM = refs[data[0][2]].db.pixel * 1e6
names = ("dist", "poni1", "poni2", "rot1", "rot2")

e_c, e_g = [], []
t0 = time.time()
for img, tp, ph, pred, sig in data:
    r1 = refs[ph].refine(img, pred, iters=400, sigma_norm=sig, trust_k=0.5)
    r2 = refs[ph].refine(img, tp, iters=400, sigma_norm=None)
    e_c.append(np.abs(r1["params"] - tp) * 1000)
    e_g.append(np.abs(r2["params"] - tp) * 1000)
e_c, e_g = np.array(e_c), np.array(e_g)

print(f"\n=== 实际工况子集 N={len(data)}（{time.time()-t0:.0f}s，像素 {PX_UM:.0f}μm）===")
print(f"{'参数':6s}{'C0精修中位':>10s}{'GT漂移中位':>10s}{'GT漂移P90':>9s}   (mm/mrad)")
for i, nm in enumerate(names):
    print(f"{nm:6s}{np.median(e_c[:, i]):10.3f}{np.median(e_g[:, i]):10.3f}"
          f"{np.percentile(e_g[:, i], 90):9.3f}")
poni_c = ((e_c[:, 1] + e_c[:, 2]) / 2 / PX_UM * 1000)
poni_g = ((e_g[:, 1] + e_g[:, 2]) / 2 / PX_UM * 1000)
print(f"\nponi 换算像素: C0精修中位 {np.median(poni_c):.2f}px  "
      f"(全体参考 1.16px)   GT漂移中位 {np.median(poni_g):.2f}px  (全体参考 2.53px)")
print(f"C0 dist<0.5mm 占比 {(e_c[:,0]<0.5).mean()*100:.0f}%  (全体 77.5%)")
