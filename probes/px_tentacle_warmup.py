# -*- coding: utf-8 -*-
"""px 触须热身：冻结骨干，只训 px 触须（12+12 参数）。

px_v1 事故机理：k>1 图在像素域长得像 k=1 近距图 → 骨干按 k=1 语义读数，
触须本应补 dist∝k 的物理平移（归一化域 +2log10(k)/1.477），但联合微调
8 epoch 只学到 ~25%（老骨干先验抢梯度），远距粗解全废。

疗法：物理解析初始化触须 dist 通道（model.py 已内置）+ 本脚本冻结骨干
单独精修触须（高学习率，数据自选精确斜率/截距/logvar）→ 存
runs/px_v2_warm.pt → train.py --init-from 低学习率联合微调。

用法：python -u -X utf8 probes/px_tentacle_warmup.py
"""
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import batch_lam, batch_px, loss_fn
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data/train_px1")
SRC = os.path.join(ROOT, "runs/base_v3lamp/best.pt")
OUT = os.path.join(ROOT, "runs/px_v2_warm.pt")
EPOCHS = 2
LR = 3e-3

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
m = ResNet18Geo(lam_cond=True, px_cond=True)
ck = torch.load(SRC, map_location="cpu", weights_only=False)
m.load_state_dict(ck["model"], strict=False)      # 骨干+λ触须来自 lamp，px=物理初始化
print(f"热身起点：px_proj_mu.dist 权重 = {m.px_proj_mu.weight[0,0].item():.4f}"
      f"（物理平均斜率 0.323）")
m.to(dev).train()

for n, p in m.named_parameters():
    p.requires_grad = "px_proj" in n
tent = [p for p in m.parameters() if p.requires_grad]
print(f"可训练参数：{sum(p.numel() for p in tent)} 个（{[n for n,_ in m.named_parameters() if 'px_proj' in n]}）")

tr = TorchFAIDataset(DATA, "train")
va = TorchFAIDataset(DATA, "val")
consts = tr.norm_consts
tl = DataLoader(tr, batch_size=48, shuffle=True, num_workers=0, drop_last=True)
opt = torch.optim.Adam(tent, lr=LR)

aux_all = va._maux
par_all = va._mparams


def val_per_k():
    m.eval()
    ks = aux_all[:, 6]
    out = {}
    vl = DataLoader(va, batch_size=48, shuffle=False, num_workers=0)
    with torch.no_grad():
        for k_gr in np.unique(ks.round(2)):
            idx = np.where(ks.round(2) == k_gr)[0][:48]
            errs = []
            for j in range(0, len(idx), 48):
                ii = idx[j:j + 48]
                xs, ys, auxs = [], [], []
                for g in va._sel[ii]:
                    x, y, a = va[int(np.searchsorted(va._sel, g))]
                    xs.append(x); ys.append(y); auxs.append(a)
                x = torch.stack(xs)[:, None].to(dev)
                y = torch.stack(ys).to(dev)
                a = torch.stack(auxs)
                mu, _ = m(x, batch_lam(a).to(dev), batch_px(a).to(dev))
                p = norm_to_param(mu.float().cpu().numpy(), consts,
                                  batch_px(a).numpy())
                gt = par_all[ii[j:j + 48]]
                errs.extend(np.abs(p[:, 0] - gt[:, 0]) * 1000)
            out[k_gr] = (np.median(errs), len(errs))
    m.train()
    return out


best = None
for ep in range(EPOCHS):
    t0, run, rn = time.time(), 0.0, 0
    for bi, (x, y, aux) in enumerate(tl):
        x = x.unsqueeze(1).to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        lam, px = batch_lam(aux).to(dev), batch_px(aux).to(dev)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            mu, lv = m(x, lam, px)
            l, _ = loss_fn(mu.float(), lv.float(), y, "nll", None)
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        run += float(l) * len(x); rn += len(x)
        if (bi + 1) % 100 == 0:
            print(f"ep{ep} {bi+1}/{len(tl)} loss={run/rn:.4f} "
                  f"w={m.px_proj_mu.weight[0,0].item():+.4f} "
                  f"b={m.px_proj_mu.bias[0].item():+.4f}", flush=True)
    r = val_per_k()
    med = np.mean([v[0] for v in r.values()])
    print(f"ep{ep} done {time.time()-t0:.0f}s 逐k中位(mm): "
          + " ".join(f"{k:.2f}:{v[0]:.0f}" for k, v in r.items()),
          f"→ 均值 {med:.0f}mm", flush=True)
    if best is None or med < best[0]:
        best = (med, {k: v[0] for k, v in r.items()})
        torch.save(dict(model=m.state_dict()), OUT)

print(f"\n最优触须：逐k中位 {best[1]} → {OUT}")
print(f"w={m.px_proj_mu.weight.detach().cpu().numpy().round(3)} "
      f"b={m.px_proj_mu.bias.detach().cpu().numpy().round(3)}")
