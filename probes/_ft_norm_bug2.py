# -*- coding: utf-8 -*-
"""用【模型自身训练时的归一化常数】解码，重新评测 v3 vs v4b。

v4/v4b 在 data/finetune_v1（meta.config.dist_range=[0.1,1.0]）上训练，
但之前统一用 train_v2 的常数（dist_range=[0.1,3.0]）解码 → 纯数学偏差，
被误判成"灾难性遗忘"。

本探针：dist 通道用 finetune_v1 常数解码，poni/rot 通道常数两套一致（先验证），
然后与 train_v2 的物理真值比。
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ft = TorchFAIDataset(os.path.join(ROOT, "data", "finetune_v1"), split="train")
v2 = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")

# ---- 先验证 poni/rot 的归一化常数两套是否一致 ----
print("=== 归一化常数一致性检查 ===")
for k in ("Hm", "Wm"):
    same = np.isclose(ft.norm_consts[k], v2.norm_consts[k], rtol=0, atol=1e-9)
    print(f"  {k}: finetune={ft.norm_consts[k]:.6f} train_v2={v2.norm_consts[k]:.6f} "
          f"-> {'一致' if same else '不一致!!'}")
for i in (1, 2, 3):
    k = f"r{'' if i==3 else i}_{'lo' if i!=2 else 'lo'}"
for tag in ("r_lo", "r_hi"):
    a, b = ft.norm_consts[tag], v2.norm_consts[tag]
    print(f"  {tag}: finetune={a} train_v2={b} -> "
          f"{'一致' if np.allclose(a,b) else '不一致!!'}")
print(f"  dist: dlo/dhi finetune=({ft.norm_consts['dlo']:.4f},{ft.norm_consts['dhi']:.4f}) "
      f"train_v2=({v2.norm_consts['dlo']:.4f},{v2.norm_consts['dhi']:.4f}) "
      f"-> {'一致' if np.isclose(ft.norm_consts['dhi'],v2.norm_consts['dhi']) else '不一致!!(核心)'}")
print(f"  peak_sigma/dw_b 等其余常数仅影响生成，不参与 6 参数归一化\n")

aux = v2._maux[v2._sel]
nvis, imax, hump = aux[:, 1], aux[:, 3], aux[:, 4]
prac = (nvis >= 6) & (imax >= 10000) & (hump < 0)
dists = v2._mparams[v2._sel][:, 0]
far = dists > 0.77
TIERS = [("全体", np.ones(len(v2._sel), bool)),
         ("实际工况", prac),
         ("远距>0.77m", far)]
PX_UM = 344.0

# (tag, ckpt, 用哪套常数解码 dist)
CKPTS = [("base_v3", "runs/base_v3/best.pt", v2.norm_consts),   # v3 用 train_v2 训，常数一致
         ("base_v4", "runs/base_v4/best.pt", ft.norm_consts),    # v4 用 finetune 训
         ("base_v4b", "runs/base_v4b/best.pt", ft.norm_consts)]   # v4b 用 finetune 训

truth = v2._mparams[v2._sel]

for tag, rel, nc in CKPTS:
    fp = os.path.join(ROOT, rel)
    if not os.path.exists(fp):
        print(f"\n===== {tag}: 缺 checkpoint, 跳过 =====")
        continue
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval().cuda()
    E = np.zeros((len(v2._sel), 6))
    t0 = time.time()
    idx_all = np.arange(len(v2._sel))
    for s in range(0, len(idx_all), 64):
        ks = idx_all[s:s + 64]
        xb = np.stack([np.log1p(np.asarray(v2._mimgs[int(v2._sel[k])],
                                           dtype=np.float32)) for k in ks])
        with torch.no_grad():
            x = torch.from_numpy(xb)[:, None].cuda()
            mu, lv = model(x)
        for j, k in enumerate(ks):
            phys = norm_to_param(mu[j].cpu().numpy().astype(np.float64), nc)
            E[k] = np.abs(phys - truth[k])
    print(f"\n===== {tag}（dist 用 {('finetune' if nc is ft.norm_consts else 'train_v2')} 常数，"
          f"{time.time()-t0:.0f}s）=====")
    print(f"{'档':10s}{'N':>5s}{'distMAE':>9s}{'dist中位':>9s}{'尾>10mm':>9s}"
          f"{'poni中位px':>11s}{'rot1中位mrad':>13s}")
    for tname, m in TIERS:
        e = E[m] * 1000
        poni_px = np.median((e[:, 1] + e[:, 2]) / 2 / PX_UM * 1000)
        print(f"{tname:10s}{m.sum():5d}{np.mean(e[:, 0]):9.2f}{np.median(e[:, 0]):9.3f}"
              f"{(e[:, 0] > 10).mean()*100:8.1f}%{poni_px:11.2f}{np.median(e[:, 3]):13.3f}")
