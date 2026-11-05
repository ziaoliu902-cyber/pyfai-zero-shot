# -*- coding: utf-8 -*-
"""标定 σ 的精修收益 A/B（实际工况档 60 张，seed=11 与历史口径可比）。

配方（全部 base_v3/last.pt 粗解起点）：
  M1  soft_prior=0.1, trust_k=1.0, 原始 σ        ← 现行默认（历史基准）
  M1c soft_prior=0.1, trust_k=1.0, σ·T（标定）    ← 唯一变量：σ 标定
  C0  trust_k=0.5, 原始 σ（硬笼基线）
  C0c trust_k=0.5, σ·T（标定紧笼）

输出：dist MAE/中位/<0.5mm、poni 中位 px、rot2 中位 mrad + M1→M1c 配对符号检验。
"""
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 60
CKPT = os.path.join(ROOT, "runs/base_v3/last.pt")

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split="test")
aux = ds._maux[ds._sel]
mask = (aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
idx = np.where(mask)[0]
sel = sorted(np.random.default_rng(11).choice(idx, min(N, len(idx)), replace=False).tolist())
print(f"实际工况档 {mask.sum()} 张，抽 {len(sel)} (seed=11)")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"])
model.eval().cuda()

import json
T = np.array([json.load(open(os.path.join(ROOT, "runs/base_v3/sigma_T.json"),
                            encoding="utf-8"))["T"][n] for n in
              ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")])
print(f"T = {T}")

refs, samples = {}, []
for k in sel:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    tp = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
        mu, lv = model(x)
        pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
        sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v3"))
    samples.append((img, tp, ph, pred, sig))

px_mm = refs[samples[0][2]].db.pixel * 1000.0
RECIPES = {
    "M1":  dict(trust_k=1.0, soft_prior=0.1, calib=False),
    "M1c": dict(trust_k=1.0, soft_prior=0.1, calib=True),
    "C0":  dict(trust_k=0.5, soft_prior=0.0, calib=False),
    "C0c": dict(trust_k=0.5, soft_prior=0.0, calib=True),
}
res = {k: [] for k in RECIPES}
t0 = time.time()
for i, (img, tp, ph, pred, sig) in enumerate(samples):
    for rn, rc in RECIPES.items():
        s = sig * T if rc["calib"] else sig
        r = refs[ph].refine(img, pred, iters=400, sigma_norm=s,
                            trust_k=rc["trust_k"], soft_prior=rc["soft_prior"])
        res[rn].append(np.abs(r["params"] - tp) * 1000)      # mm / mrad
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(samples)}  {time.time()-t0:.0f}s", flush=True)

print(f"\n像素 {px_mm*1000:.0f}μm；全部 {time.time()-t0:.0f}s\n")
hdr = f"{'配方':5s}{'distMAE':>9s}{'dist中位':>9s}{'<0.5mm':>8s}{'poni中位px':>11s}{'poni_p75':>9s}{'rot2中位':>9s}"
print(hdr)
out = {}
for rn, e in res.items():
    e = np.array(e)
    poni_px = (e[:, 1] + e[:, 2]) / 2 / px_mm
    row = dict(mae=e[:, 0].mean(), med=np.median(e[:, 0]),
               hit=(e[:, 0] < 0.5).mean() * 100,
               poni_med=np.median(poni_px), poni_p75=np.percentile(poni_px, 75),
               rot2=np.median(e[:, 4]))
    out[rn] = row
    print(f"{rn:5s}{row['mae']:9.3f}{row['med']:9.3f}{row['hit']:7.1f}%"
          f"{row['poni_med']:11.2f}{row['poni_p75']:9.2f}{row['rot2']:9.3f}")

# 配对符号检验 M1 → M1c（dist）
e1 = np.array(res["M1"])[:, 0]
e2 = np.array(res["M1c"])[:, 0]
w = int((e2 < e1).sum())
p = 2 * (1 - 0.5 - 0.5 * __import__("scipy.stats", fromlist=["norm"]).norm.cdf(
    abs(w - len(e1) / 2) / np.sqrt(len(e1) / 4))) if len(e1) else 1.0
try:
    from scipy.stats import binomtest
    p = binomtest(w, len(e1), 0.5).pvalue
except Exception:
    pass
print(f"\n配对符号检验 M1→M1c (dist)：M1c 胜 {w}/{len(e1)}，p={p:.4f}")
np.savez_compressed(os.path.join(ROOT, "probes", "results",
                                 f"sigma_ab_{time.strftime('%Y%m%d_%H%M%S')}.npz"),
                    **{k: np.array(v) for k, v in res.items()})
print("落盘 probes/results/sigma_ab_*.npz")
