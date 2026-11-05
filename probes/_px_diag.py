# -*- coding: utf-8 -*-
"""px_v1 宽视域失准诊断：sim 域逐 k 评估 + 触须权重 + 合成 k=2.43 直测。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio

from torchfai.config import SimConfig
from torchfai.dataset import TorchFAIDataset, norm_to_param, param_to_norm
from torchfai.model import ResNet18Geo
from torchfai.simulator import get_scaled_detector, synth_sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(os.path.join(ROOT, "runs/px_v1/best.pt"),
                map_location="cpu", weights_only=False)
m = ResNet18Geo(lam_cond=True, px_cond=True).eval()
m.load_state_dict(ck["model"])
m.to(dev)
consts = TorchFAIDataset(os.path.join(ROOT, "data/train_px1"), "val").norm_consts

print("== 1. px 触须权重（零初始化后学到多少）==")
for nm in ("px_proj_mu", "px_proj_logvar"):
    w = getattr(m, nm).weight.detach()
    print(f"   {nm}: |w|₁={w.abs().sum():.4f} max={w.abs().max():.4f}")

print("== 2. sim 域 val 逐 k 粗解 dist MAE ==")
ds = TorchFAIDataset(os.path.join(ROOT, "data/train_px1"), "val")
gidx = ds._sel
aux = np.load(os.path.join(ROOT, "data/train_px1/aux.npy"))[gidx]
par = np.load(os.path.join(ROOT, "data/train_px1/params.npy"))[gidx]
ks = aux[:, 6]
errs = {k: [] for k in np.unique(ks.round(2))}
with torch.no_grad():
    for i in range(0, len(ds), 4):          # 抽 1/4 快测
        x = torch.from_numpy(np.log1p(
            np.asarray(ds._mimgs[gidx[i]]).astype(np.float32)))[None, None].to(dev)
        mu, _ = m(x, torch.tensor([aux[i, 5]], device=dev),
                  torch.tensor([ks[i]], device=dev))
        p = norm_to_param(mu.float().cpu().numpy(), consts, float(ks[i]))[0]
        errs[ks[i].round(2)].append(abs(p[0] - par[i, 0]) * 1000)
for k, e in errs.items():
    e = np.array(e)
    print(f"   k={k}: n={len(e)} 中位={np.median(e):7.1f}mm "
          f"P90={np.percentile(e, 90):8.1f}mm")

print("== 3. 合成 k=2.43 直测（ceria 远距 0.45m）==")
cfg = SimConfig()
cfg.wavelength = 9.72386e-11
db = get_scaled_detector("Pilatus_1M", 2, 2.425)
Hm, Wm = db.extent
rng = np.random.default_rng(5)
g = dict(dist=0.45, poni1=Hm * 0.5, poni2=Wm * 0.5, rot1=0.0, rot2=0.0, rot3=0.0)
s = synth_sample(rng, cfg, db, phase_idx=0, pixel_scale=2.425, geom=g)
with torch.no_grad():
    x = torch.from_numpy(np.log1p(s["img"].astype(np.float32)))[None, None].to(dev)
    mu, _ = m(x, torch.tensor([9.72386], device=dev),
              torch.tensor([2.425], device=dev))
p = norm_to_param(mu.float().cpu().numpy(), consts, 2.425)[0]
print(f"   GT=450mm 预测={p[0]*1000:.1f}mm Δ={(p[0]-0.45)*1000:+.1f}mm "
      f"poniΔ=({(p[1]-Hm*0.5)/db.pixel:+.0f},{(p[2]-Wm*0.5)/db.pixel:+.0f})px")

print("== 4. 同一合成样本喂 k=1（错误条件）对照 ==")
with torch.no_grad():
    mu, _ = m(x, torch.tensor([9.72386], device=dev),
              torch.tensor([1.0], device=dev))
p1 = norm_to_param(mu.float().cpu().numpy(), consts, 1.0)[0]
print(f"   条件错标 k=1 时预测={p1[0]*1000:.1f}mm（应明显跑偏→证明触须在起作用）")

print("== 5. 真实 6M 150mm 宽视域（复现考卷 A 输入）==")
from real_refine_v6 import to_nn_input_wide, nn_coarse
img = fabio.open(os.path.join(ROOT, "data/real/MX_ceria/ceria_150_1_0001.cbf")).data
wide, k, extra = to_nn_input_wide(img)
pr, _ = nn_coarse(m, dev, consts, wide, 0.972386, k)
print(f"   k={k:.3f} 预测={pr[0]*1000:.1f}mm（GT=150）")
print(f"   帧统计: max={wide.max():.0f} 均值={wide.mean():.1f} "
      f"零占比={(wide==0).mean():.3f}")
