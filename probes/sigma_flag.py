# -*- coding: utf-8 -*-
"""坏样本三道保险之②验证：NN 自报 σ 能否自动标记"卡住"的样本。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 100

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

rng = np.random.default_rng(11)
ks = sorted(rng.choice(len(ds._sel), N, replace=False).tolist())
sig_d, err_a, err_b, losses = [], [], [], []
refs = {}
for k in ks:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    tp = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
        mu, lv = model(x)
        mu_np = mu[0].cpu().numpy().astype(np.float64)
        sg = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
    pred = norm_to_param(mu_np, ds.norm_consts)
    # σ 转物理单位：norm 空间扰动 ±σ → 参数变化
    sig_phys = np.abs(norm_to_param(mu_np + sg, ds.norm_consts)
                      - norm_to_param(mu_np - sg, ds.norm_consts)) / 2
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
    r = refs[ph].refine(img, pred, iters=400, sigma_norm=sg, trust_k=0.5)
    err_b.append(abs(pred[0] - tp[0]) * 1000)
    err_a.append(abs(r["params"][0] - tp[0]) * 1000)
    sig_d.append(sig_phys[0] * 1000)
    losses.append(float(r["loss"]))

err_b, err_a, sig_d, losses = map(np.array, (err_b, err_a, sig_d, losses))
stuck = err_a > 5.0        # 精修后仍 >5mm 视为卡住
print(f"N={N}  卡住(>5mm) {stuck.sum()} 例 ({stuck.mean()*100:.0f}%)")
print(f"卡住样本: 精修后误差中位 {np.median(err_a[stuck]):.1f}mm  (初值误差中位 {np.median(err_b[stuck]):.1f}mm)")
print(f"未卡样本: 精修后误差中位 {np.median(err_a[~stuck]):.2f}mm")
print()
print("物理引擎最终 Poisson 损失能否分离卡住样本？（不用真值的考卷分）")
lo_s, lo_g = losses[stuck], losses[~stuck]
print(f"  卡住样本 loss: {lo_s.min():.3f} ~ {lo_s.max():.3f} (中位 {np.median(lo_s):.3f})")
print(f"  正常样本 loss: {lo_g.min():.3f} ~ {lo_g.max():.3f} (中位 {np.median(lo_g):.3f})")
# 按相内相对排位打分更公平（不同相基准不同）：z-score within phase 太复杂，
# 直接看绝对分离度:
thr = (np.sort(lo_s).min() + np.sort(lo_g)[:80].max()) / 2 if len(lo_s) else 0
sep = lo_s.min() - np.percentile(lo_g, 95)
print(f"  分离度: 卡住最低分 - 正常95%分位 = {sep:+.3f}  "
      f"({'完全可分' if lo_s.min() > np.percentile(lo_g, 95) else '部分重叠'})")
for T in sorted(set(np.round(np.percentile(lo_g, [90, 95, 99]), 3))):
    flag = losses > T
    tp_ = (flag & stuck).sum(); fp = (flag & ~stuck).sum()
    print(f"  阈值 loss>{T:.3f}: 抓获 {tp_}/{stuck.sum()} | 误报 {fp}/{(~stuck).sum()}")

