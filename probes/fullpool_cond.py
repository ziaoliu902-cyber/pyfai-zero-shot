# -*- coding: utf-8 -*-
"""实际工况档全量 394：条件 σ 标定 × 信任域/先验解耦（三代配方叠加）。

在 fullpool_split.py（σ 用在哪）之上再加一层：**σ 的标定系数按档分组**
（probes/sigma_cond.py 产出 runs/base_v3/sigma_T_cond.json）。

三代配方演进：
  1. M1     = refine(σ 笼 + σ 先验)                     旧默认
  2. M1cd   = refine_dual(σ·T_glob 笼 + σ·T_glob 先验)   10-27 继任者候选
  3. M2cd   = refine_dual(σ 笼 + σ·T_glob 先验)         信任域/先验解耦
  4. M3cd   = refine_dual(σ 笼 + σ·T_cond 先验)         再叠加条件标定
  5. M3cdx  = refine_dual(1.5σ 笼 + σ·T_cond 先验)      笼子病兜底

语义（为什么这样分）：
  信任域 = 搜索空间：要够宽，装得下真值 → 用原 σ（或放宽）
  软先验 = 统计权重：要用【标定后】σ，误差尺度才正确 → 用 σ·T_cond[组]

本脚本只算 4/5 两条新配方（3/1/2 已由 fullpool_split.py 落盘，按 index 对齐合并）。
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
DATA = os.path.join(ROOT, "data/train_v3")
PARAM_NAMES = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
ITERS = 400
TRUST_K = 1.0
SOFT = 0.1
ADOPT_NATS = 1.0


def main():
    ds = TorchFAIDataset(DATA, split="test")
    aux = ds._maux[ds._sel]
    mask = (aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
    sel = np.where(mask)[0]
    print(f"实际工况档全池 N={len(sel)}", flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"])
    model.eval().cuda()

    Tg = json.load(open(os.path.join(ROOT, "runs/base_v3/sigma_T.json"),
                        encoding="utf-8"))["T"]
    Tc = json.load(open(os.path.join(ROOT, "runs/base_v3/sigma_T_cond.json"),
                        encoding="utf-8"))["T_cond"]
    T_glob = np.array([Tg[n] for n in PARAM_NAMES])
    # 本评测池全部属于 "actual" 档（抽样判据即 actual）
    T_act = np.array([Tc["actual"][n] for n in PARAM_NAMES])
    print(f"T_glob = {np.round(T_glob, 3).tolist()}")
    print(f"T_act  = {np.round(T_act, 3).tolist()}", flush=True)

    refs = {}
    samples = []
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
            refs[ph] = DifferentiableRefiner(ph, DATA)
        samples.append((img, tp, ph, pred, sig))

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_fp = os.path.join(OUTDIR, f"fullpool_cond_{ts}.csv")
    fh = open(csv_fp, "w", newline="", encoding="utf-8")
    fw = csv.writer(fh)
    hdr = ["i", "phase"]
    for r in ("m3cd", "m3cdx"):
        for k in ("dist", "poni1", "poni2", "rot2"):
            hdr.append(f"{r}_{k}")
    hdr += ["adopt_m3cd", "adopt_m3cdx",
            "pure_m3cd", "pure_m3cdx", "pure_B"]
    fw.writerow(hdr)

    t0 = time.time()
    rows = []
    for i, (img, tp, ph, pred, sig) in enumerate(samples):
        ref = refs[ph]
        sigT_act = sig * T_act

        # A 轨：解耦（原始 σ 笼）+ 条件标定先验
        r_a = ref.refine(img, pred, iters=ITERS, sigma_norm=sig,
                         trust_k=TRUST_K, soft_prior=SOFT, sigma_prior=sigT_act)
        # A 轨：解耦 + 笼放宽 1.5×
        r_ax = ref.refine(img, pred, iters=ITERS, sigma_norm=sig * 1.5,
                          trust_k=TRUST_K, soft_prior=SOFT, sigma_prior=sigT_act)
        # B 轨（无笼无先验，确定性）
        r_b = ref.refine(img, pred, iters=ITERS, sigma_norm=None, soft_prior=0.0)

        ad_a = r_b["pure_loss"] < r_a["pure_loss"] - ADOPT_NATS
        ad_ax = r_b["pure_loss"] < r_ax["pure_loss"] - ADOPT_NATS
        p_a = r_b["params"] if ad_a else r_a["params"]
        p_ax = r_b["params"] if ad_ax else r_ax["params"]

        e = lambda p: np.abs(p - tp) * 1000
        row = [i, ph]
        for p_ in (p_a, p_ax):
            v = e(p_)
            row += [f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}", f"{v[4]:.4f}"]
        row += [int(ad_a), int(ad_ax),
                f"{r_a['pure_loss']:.4f}", f"{r_ax['pure_loss']:.4f}",
                f"{r_b['pure_loss']:.4f}"]
        fw.writerow(row)
        rows.append(row)
        if (i + 1) % 20 == 0:
            fh.flush()
            el = time.time() - t0
            print(f"  {i+1}/{len(samples)}  {el:.0f}s "
                  f"(ETA {el/(i+1)*(len(samples)-i-1):.0f}s)", flush=True)
    fh.close()

    arr = np.array([[float(x) for x in r[2:10]] for r in rows])
    ad = np.array([[int(rows[i][10]), int(rows[i][11])] for i in range(len(rows))])
    print(f"\n=== {len(rows)} 张（{time.time()-t0:.0f}s）===")
    for j, nm in enumerate(("m3cd", "m3cdx")):
        e = arr[:, j * 4:(j + 1) * 4]
        pp = (e[:, 1] + e[:, 2]) / 2 / 0.344
        print(f"{nm:7s} distMAE {e[:,0].mean():.3f}  中位 {np.median(e[:,0]):.3f}  "
              f"<0.5mm {(e[:,0]<0.5).mean()*100:.1f}%  >5mm {int((e[:,0]>5).sum())}  "
              f"poni中位 {np.median(pp):.2f}px  rot2中位 {np.median(e[:,3]):.3f}  "
              f"采纳 {ad[:,j].sum()}/{len(rows)}")
    print(f"落盘 {csv_fp}")


if __name__ == "__main__":
    main()
