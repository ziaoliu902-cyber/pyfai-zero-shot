# -*- coding: utf-8 -*-
"""精修实测 v2：守门策略 + 饱和像素诊断。

用法：python refine_probe.py [N] [iters] [--sat]
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if len(args) > 0 else 150
ITERS = int(args[1]) if len(args) > 1 else 300
INC_SAT = "--sat" in sys.argv
print(f"N={N} iters={ITERS} include_sat={INC_SAT}")

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

refs: dict[str, DifferentiableRefiner] = {}

def get_refiner(ph: str) -> DifferentiableRefiner:
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
    return refs[ph]

ERR = np.zeros((N, 6))     # NN 粗解
ERR2 = np.zeros((N, 6))    # 精修后（无守门）
ERR3 = np.zeros((N, 6))    # 精修后（守门：loss<loss_init 才接受）
DIST = np.zeros(N)
SATF = np.zeros(N)
GUARD_REJ = []
TIMES = []

t_all = time.time()
for k in range(N):
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    true_p = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]

    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
        mu, lv = model(x)
        pred_phys = norm_to_param(mu[0].cpu().numpy().astype(np.float64),
                                  ds.norm_consts)
        sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)

    r = get_refiner(ph).refine(img, pred_phys, iters=ITERS, include_sat=INC_SAT,
                               sigma_norm=sig, trust_k=0.5)
    TIMES.append(r["sec"])
    ERR[k] = np.abs(pred_phys - true_p) * 1000.0
    ERR2[k] = np.abs(r["params"] - true_p) * 1000.0
    accept = r["loss"] < r["loss_init"]
    if accept:
        ERR3[k] = ERR2[k]
    else:
        ERR3[k] = ERR[k]
        GUARD_REJ.append((k, ph, ERR[k, 0], ERR2[k, 0]))
    DIST[k] = true_p[0]
    SATF[k] = float((img >= 65535).mean())
    if (k + 1) % 25 == 0:
        print(f"  {k+1}/{N}  {time.time()-t_all:.0f}s  "
              f"dist: 粗{ERR[:k+1,0].mean():.2f} 精修{ERR2[:k+1,0].mean():.2f} "
              f"守门{ERR3[:k+1,0].mean():.2f}mm")

names = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
print(f"\n=== {N} 张（{time.time()-t_all:.0f}s，单张 {np.mean(TIMES):.2f}s）===")
print(f"{'参数':8s}{'NN粗解':>10s}{'精修':>10s}{'守门后':>10s}")
for i, nm in enumerate(names):
    print(f"{nm:8s}{ERR[:,i].mean():10.3f}{ERR2[:,i].mean():10.3f}"
          f"{ERR3[:,i].mean():10.3f}")

for lo, hi, tag in ((0, 0.39, "近距"), (0.39, 0.77, "中距"), (0.77, 9, "远距")):
    m = (DIST >= lo) & (DIST < hi)
    if m.sum():
        print(f"{tag} N={m.sum():3d}  dist 粗{ERR[m,0].mean():7.2f} "
              f"精修{ERR2[m,0].mean():7.2f} 守门{ERR3[m,0].mean():7.2f}mm "
              f"饱和率{SATF[m].mean()*100:.2f}%")

# 饱和率与近距误差的相关性
m_near = DIST < 0.39
if m_near.sum() > 10:
    c = np.corrcoef(SATF[m_near], ERR2[m_near, 0])[0, 1]
    print(f"近距: 饱和率~精修误差 相关系数 r={c:+.2f}  平均饱和率={SATF[m_near].mean()*100:.2f}%")

print(f"\n守门拒绝 {len(GUARD_REJ)} 例:")
for k, ph, b, a in GUARD_REJ[:12]:
    print(f"  #{k} {ph} dist={DIST[k]:.2f}m  粗{b:.1f} 精修{a:.1f}mm")
