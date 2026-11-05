# -*- coding: utf-8 -*-
"""判别实验：GT 漂移是"系统偏置"还是"nuisance 吸收"？

同 20 样本三组：
  G1: GT 起点、无笼、默认 nuisance 自由度      → 带符号漂移 + 终点 loss
  G2: NN 起点、0.5σ 笼（=C0）                 → 带符号残差 + 终点 loss
  G3: GT 起点、无笼、环宽收紧(0.003,0.011)     → 漂移是否缩水
若 G1 终点 loss < G2 终点 loss 且 G1 漂移方向一致 → 似然真心偏置；
若 G3 漂移缩水 → 环宽自由度在吸收几何信号。
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
NGT = 20

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

rng = np.random.default_rng(7)
ks = sorted(rng.choice(len(ds._sel), 60, replace=False).tolist())[:NGT]
data, refs, phs = [], {}, []
for k in ks:
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
    phs.append(ph)

names = ("dist", "poni1", "poni2", "rot1", "rot2")


def signed(fn):
    out = []
    for img, tp, ph, pred, sig in data:
        r = fn(refs[ph], img, pred, sig, tp)
        out.append(((r["params"] - tp) * 1000, r["loss"]))
    d = np.array([o[0] for o in out])
    ls = np.array([o[1] for o in out])
    return d, ls


t0 = time.time()
d1, l1 = signed(lambda r, i, p, s, tp: r.refine(i, tp, iters=400,
                                                 sigma_norm=None))
print(f"G1 GT无笼     loss终={l1.mean():.4f}  {time.time()-t0:.0f}s", flush=True)
d2, l2 = signed(lambda r, i, p, s, tp: r.refine(i, p, iters=400, sigma_norm=s,
                                                 trust_k=0.5))
print(f"G2 NN+0.5σ笼  loss终={l2.mean():.4f}  {time.time()-t0:.0f}s", flush=True)
d3, l3 = signed(lambda r, i, p, s, tp: r.refine(i, tp, iters=400,
                                                 sigma_norm=None,
                                                 sigma_range=(0.003, 0.011)))
print(f"G3 GT+环宽紧  loss终={l3.mean():.4f}  {time.time()-t0:.0f}s", flush=True)

print(f"\n样本相分布: {dict((p, phs.count(p)) for p in set(phs))}")
print(f"\n{'':10s}{'漂移均值(带符号)':>12s}  {'中位|漂移|':>8s}  G3同项中位|漂移|")
for i, nm in enumerate(names):
    print(f"{nm:10s}{d1[:, i].mean():+12.3f}  {np.median(np.abs(d1[:, i])):8.3f}"
          f"  {np.median(np.abs(d3[:, i])):8.3f}")
print(f"\nG2(NN起点+笼) 中位|残差|: "
      + "  ".join(f"{nm}={np.median(np.abs(d2[:, i])):.3f}"
                  for i, nm in enumerate(names)))
print(f"这20张 dist 分布(m): "
      f"{np.percentile([t[1][0] for t in data], [10,50,90]).round(3)}")
print(f"loss 对比: GT起点终loss 平均 {l1.mean():.4f} vs NN起点终loss {l2.mean():.4f}"
      f"  (G1更小→似然真心偏离GT；G2更小或相当→优化/简并问题)")
print(f"逐样本 G1<G2 占比: {(l1 < l2).mean()*100:.0f}%")
