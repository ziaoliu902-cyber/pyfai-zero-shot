# -*- coding: utf-8 -*-
"""软 MAP 先验 A/B（落盘版）——与 _map_prior_ab.py 协议逐位一致。

与原脚本的差异仅在于"结果落盘"：
  - probes/results/map_prior_ab_<ts>.csv   汇总表（每配置每档一行）
  - probes/results/map_prior_ab_<ts>.npz   逐样本原始误差
  - probes/results/map_prior_ab_<ts>.log   完整 stdout
  - 每样本进度打印，便于中途判断是否卡死

协议（勿改，否则与历史口径 3.34px/7.7mrad 不可比）：
  - test 档 = data/train_v2 split=test（seed=0 默认切分）
  - 实际工况档：n_vis>=6 且 peak_scale>=10000 且 bs_r<0（无挡光盘），rng seed=11 取 30
  - 全体档：全 test 集 rng seed=7 取 30
  - 配方：C0 硬笼 0.5σ；M1/M2/M3 硬笼放宽 1σ 作保险丝 + 软先验 w=0.1/1/10
  - 精修 iters=400，传 NN 自报 sigma_norm
"""
import csv
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
OUTDIR = os.path.join(ROOT, "probes", "results")
os.makedirs(OUTDIR, exist_ok=True)
TS = time.strftime("%Y%m%d_%H%M%S")
CSV_FP = os.path.join(OUTDIR, f"map_prior_ab_{TS}.csv")
NPZ_FP = os.path.join(OUTDIR, f"map_prior_ab_{TS}.npz")

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

CONFIGS = [
    ("C0 硬笼0.5σ", dict(trust_k=0.5)),
    ("M1 软w=0.1", dict(trust_k=1.0, soft_prior=0.1)),
    ("M2 软w=1", dict(trust_k=1.0, soft_prior=1.0)),
    ("M3 软w=10", dict(trust_k=1.0, soft_prior=10.0)),
]


def load_samples(sel):
    out, refs = [], {}
    for k in sel:
        g = int(ds._sel[k])
        img = np.asarray(ds._mimgs[g])
        tp = ds._mparams[g].astype(np.float64)
        ph = PHASE_NAMES[int(ds._maux[g][0])]
        with torch.no_grad():
            x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
            mu, lv = model(x)
            pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64),
                                 ds.norm_consts)
            sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
        if ph not in refs:
            refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
        out.append((img, tp, ph, pred, sig))
    return out, refs


aux = ds._maux[ds._sel]
prac_idx = np.where((aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0))[0]
rng_p = np.random.default_rng(11)
sel_p = sorted(rng_p.choice(prac_idx, 30, replace=False).tolist())
rng_a = np.random.default_rng(7)
sel_a = sorted(rng_a.choice(len(ds._sel), 30, replace=False).tolist())

P, refs_p = load_samples(sel_p)
A, refs_a = load_samples(sel_a)
PX_UM = refs_p[P[0][2]].db.pixel * 1e6

rows = []
raw = {}

print(f"实际工况档候选 {len(prac_idx)} 张，取 {len(sel_p)}；全体档取 {len(sel_a)}")
print(f"像素 {PX_UM:.1f} μm   输出前缀 {CSV_FP}", flush=True)

for tier, data, refs in (("实际工况档(30)", P, refs_p), ("全体档(30)", A, refs_a)):
    print(f"\n===== {tier}  像素{PX_UM:.0f}μm =====", flush=True)
    print(f"{'配置':14s}{'dist中位':>9s}{'<0.5mm':>7s}{'poni中位px':>10s}"
          f"{'rot1中位':>9s}{'rot2中位':>9s}{'卡住':>5s}", flush=True)
    for tag, kw in CONFIGS:
        t0 = time.time()
        errs = []
        for i, (img, tp, ph, pred, sig) in enumerate(data):
            r = refs[ph].refine(img, pred, iters=400, sigma_norm=sig, **kw)
            errs.append(np.abs(r["params"] - tp) * 1000)
            if (i + 1) % 10 == 0:
                print(f"  [{tag}] {i+1}/{len(data)}  {time.time()-t0:.0f}s", flush=True)
        e = np.array(errs)
        raw[f"{tier}|{tag}"] = e
        poni_px = np.median((e[:, 1] + e[:, 2]) / 2 / PX_UM * 1000)
        row = dict(tier=tier, config=tag, dist_med=float(np.median(e[:, 0])),
                   dist_lt05pct=float((e[:, 0] < 0.5).mean() * 100),
                   poni_med_px=float(poni_px),
                   rot1_med=float(np.median(e[:, 3])), rot2_med=float(np.median(e[:, 4])),
                   stuck_pct=float((e[:, 0] > 5).mean() * 100), sec=round(time.time() - t0, 1))
        rows.append(row)
        print(f"{tag:14s}{row['dist_med']:9.3f}{row['dist_lt05pct']:6.0f}%"
              f"{row['poni_med_px']:10.2f}{row['rot1_med']:9.3f}"
              f"{row['rot2_med']:9.3f}{row['stuck_pct']:4.0f}%  {row['sec']:.0f}s",
              flush=True)
        with open(CSV_FP, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

np.savez_compressed(NPZ_FP, **raw)
print(f"\nCSV → {CSV_FP}\nNPZ → {NPZ_FP}")
print("ALL DONE")
