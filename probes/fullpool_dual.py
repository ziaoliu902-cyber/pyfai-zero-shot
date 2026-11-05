# -*- coding: utf-8 -*-
"""实际工况档全量 394 张：M1（现行默认）vs M1cd（标定σ+双轨）论文终版数字。

- 同卷 train_v3 test 实际工况档全池（394），粗解 base_v3/last.pt
- M1  = refine(σ原, trust_k=1.0, soft_prior=0.1)      ← 10-26 全量基准复跑
- M1cd = refine_dual(σ·T, trust_k=1.0, soft_prior=0.1, adopt_nats=1.0)
- 逐行 flush 落盘 CSV（中断可增量用），结束打印汇总 + 配对符号检验
"""
import csv
import json
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
OUTDIR = os.path.join(ROOT, "probes", "results")
CKPT = os.path.join(ROOT, "runs/base_v3/last.pt")

ds = TorchFAIDataset(os.path.join(ROOT, "data/train_v3"), split="test")
aux = ds._maux[ds._sel]
mask = (aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
sel = np.where(mask)[0]
print(f"实际工况档全池 N={len(sel)}")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"])
model.eval().cuda()

T = np.array([json.load(open(os.path.join(ROOT, "runs/base_v3/sigma_T.json"),
                            encoding="utf-8"))["T"][n] for n in
              ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")])

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
ts = time.strftime("%Y%m%d_%H%M%S")
csv_fp = os.path.join(OUTDIR, f"fullpool_dual_{ts}.csv")
fh = open(csv_fp, "w", newline="", encoding="utf-8")
fw = csv.writer(fh)
fw.writerow(["i", "phase", "adopt",
             "m1_dist", "m1cd_dist", "m1_poni1", "m1cd_poni1",
             "m1_poni2", "m1cd_poni2", "m1_rot2", "m1cd_rot2",
             "pure_A", "pure_B"])

t0 = time.time()
R = {"M1": [], "M1cd": []}
adopts = 0
for i, (img, tp, ph, pred, sig) in enumerate(samples):
    a = refs[ph].refine(img, pred, iters=400, sigma_norm=sig,
                        trust_k=1.0, soft_prior=0.1)
    d = refs[ph].refine_dual(img, pred, sigma_norm=sig * T, trust_k=1.0,
                             soft_prior=0.1, iters=400, adopt_nats=1.0)
    ea = np.abs(a["params"] - tp) * 1000
    ed = np.abs(d["params"] - tp) * 1000
    R["M1"].append(ea)
    R["M1cd"].append(ed)
    adopts += int(d["dual_adopt"])
    fw.writerow([i, ph, int(d["dual_adopt"]),
                 f"{ea[0]:.4f}", f"{ed[0]:.4f}",
                 f"{ea[1]:.4f}", f"{ed[1]:.4f}",
                 f"{ea[2]:.4f}", f"{ed[2]:.4f}",
                 f"{ea[4]:.4f}", f"{ed[4]:.4f}",
                 f"{a['pure_loss']:.4f}", f"{d['pure_B']:.4f}"])
    if (i + 1) % 20 == 0:
        fh.flush()
        el = time.time() - t0
        print(f"  {i+1}/{len(samples)}  采纳 {adopts}  "
              f"{el:.0f}s (ETA {el/(i+1)*(len(samples)-i-1):.0f}s)", flush=True)
fh.close()

e1 = np.array(R["M1"])
e2 = np.array(R["M1cd"])
print(f"\n=== 全量 {len(e1)} 张（{time.time()-t0:.0f}s）采纳率 {adopts}/{len(e1)} ===")
print(f"{'配方':6s}{'distMAE':>9s}{'dist中位':>9s}{'<0.5mm':>8s}"
      f"{'poni中位px':>11s}{'poni_p75':>9s}{'poni_p90':>9s}{'rot2中位':>9s}{'>5mm':>5s}")
for nm, e in (("M1", e1), ("M1cd", e2)):
    poni_px = (e[:, 1] + e[:, 2]) / 2 / px_mm
    print(f"{nm:6s}{e[:,0].mean():9.3f}{np.median(e[:,0]):9.3f}"
          f"{(e[:,0]<0.5).mean()*100:7.1f}%{np.median(poni_px):11.2f}"
          f"{np.percentile(poni_px,75):9.2f}{np.percentile(poni_px,90):9.2f}"
          f"{np.median(e[:,4]):9.3f}{(e[:,0]>5).sum():5d}")

for col, nm in ((0, "dist"), (4, "rot2")):
    w = int((e2[:, col] < e1[:, col]).sum())
    try:
        from scipy.stats import binomtest
        p = binomtest(w, len(e1), 0.5).pvalue
    except Exception:
        p = float("nan")
    print(f"配对符号检验 M1→M1cd ({nm})：胜 {w}/{len(e1)}  p={p:.2e}")

# 误伤检查
d = e2[:, 0] - e1[:, 0]
print(f"\nM1cd vs M1 dist 逐样本差：中位 {np.median(d):+.3f}mm，"
      f"变差>1mm {(d>1).sum()} 张，变好>1mm {(d<-1).sum()} 张")
print(f"落盘 {csv_fp}")
