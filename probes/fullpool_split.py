# -*- coding: utf-8 -*-
"""实际工况档全量 394：σ 用在哪？——信任域 vs 软先验解耦的 A/B。

动机（10-27 继任者交接后复核发现）：
  fullpool_dual（394 张全池）里 M1cd = refine_dual(σ·T, trust_k=1.0, soft_prior=0.1)
  确实显著优于 M1 = refine(σ, ...)，但**唯一的误伤样本 #321（AgBh）**：
      adopt=0（双轨没采纳 B），M1 dist=0.4962mm，M1cd dist=11.7795mm
  两者 A 轨的唯一差别就是 σ·T。σ·T（dist T=0.317）把**信任域半径收窄 3.2 倍**
  → 真值在笼外 → 优化走不动 → 误差暴涨。
  根因：**σ 承担了两个语义不同的角色**——
      1) 信任域半径 = 搜索范围：要够宽（装得下真值）
      2) 软先验权重 = 统计尺度：要用标定后的 σ（误差尺度才正确）
  标定 σ 直接乘进信任域，等于把 1 也一起收窄了。

配方（同一批图、同一粗解、同 iters/trust_k/soft_prior）：
  M1    = refine(σ)                                 现行默认
  M1cd  = refine_dual(σ·T 笼)                       10-27 继任者的候选
  M2cd  = refine_dual(σ 笼 + σ·T 先验)              ← 本次要验证的解耦配方
  M2cd+ = 同 M2cd，但 A 轨 σ 笼放宽到 1.5×（笼子病兜底）
  B     = refine(无笼无先验)                        双轨的 B 轨（确定性，配方间共用）

逐行 flush 落盘 CSV（可增量续跑），结束打印终版表 + 配对检验。
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
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if limit:
        sel = sel[:limit]
    print(f"实际工况档全池 N={len(sel)}" + (f"（限量 {limit}）" if limit else ""), flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"])
    model.eval().cuda()

    T = np.array([json.load(open(os.path.join(ROOT, "runs/base_v3/sigma_T.json"),
                                encoding="utf-8"))["T"][n] for n in PARAM_NAMES])
    print(f"σ 标定 T = {np.round(T, 3).tolist()}", flush=True)

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
    csv_fp = os.path.join(OUTDIR, f"fullpool_split_{ts}.csv")
    fh = open(csv_fp, "w", newline="", encoding="utf-8")
    fw = csv.writer(fh)
    hdr = ["i", "phase"]
    for r in ("nn", "m1", "m1cd", "m2cd", "m2cdx"):
        for k in ("dist", "poni1", "poni2", "rot2"):
            hdr.append(f"{r}_{k}")
    hdr += ["adopt_m1cd", "adopt_m2cd", "adopt_m2cdx",
            "pure_m1", "pure_m1cd", "pure_m2cd", "pure_m2cdx", "pure_B"]
    fw.writerow(hdr)

    t0 = time.time()
    rows = []
    for i, (img, tp, ph, pred, sig) in enumerate(samples):
        ref = refs[ph]
        sigT = sig * T

        # --- M1: 现行默认（σ 笼 + σ 先验）---
        r_m1 = ref.refine(img, pred, iters=ITERS, sigma_norm=sig,
                          trust_k=TRUST_K, soft_prior=SOFT)
        # --- A 轨（标定 σ 进笼）---
        r_ac = ref.refine(img, pred, iters=ITERS, sigma_norm=sigT,
                          trust_k=TRUST_K, soft_prior=SOFT)
        # --- A 轨（解耦：原 σ 笼 + 标定 σ 先验）---
        r_ar = ref.refine(img, pred, iters=ITERS, sigma_norm=sig,
                          trust_k=TRUST_K, soft_prior=SOFT, sigma_prior=sigT)
        # --- A 轨（解耦 + 笼放宽 1.5×）---
        r_ax = ref.refine(img, pred, iters=ITERS, sigma_norm=sig * 1.5,
                          trust_k=TRUST_K, soft_prior=SOFT, sigma_prior=sigT)
        # --- B 轨（无笼无先验，配方间共用，确定性）---
        r_b = ref.refine(img, pred, iters=ITERS, sigma_norm=None, soft_prior=0.0)

        e = lambda p: np.abs(p - tp) * 1000

        # 双轨判定：纯似然好 >1 nat 才采纳 B
        ad_m1cd = r_b["pure_loss"] < r_ac["pure_loss"] - ADOPT_NATS
        ad_m2cd = r_b["pure_loss"] < r_ar["pure_loss"] - ADOPT_NATS
        ad_m2cdx = r_b["pure_loss"] < r_ax["pure_loss"] - ADOPT_NATS

        p_m1cd = r_b["params"] if ad_m1cd else r_ac["params"]
        p_m2cd = r_b["params"] if ad_m2cd else r_ar["params"]
        p_m2cdx = r_b["params"] if ad_m2cdx else r_ax["params"]

        ew = e
        row = [i, ph]
        for p_ in (pred, r_m1["params"], p_m1cd, p_m2cd, p_m2cdx):
            v = ew(p_)
            row += [f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}", f"{v[4]:.4f}"]
        row += [int(ad_m1cd), int(ad_m2cd), int(ad_m2cdx),
                f"{r_m1['pure_loss']:.4f}", f"{r_ac['pure_loss']:.4f}",
                f"{r_ar['pure_loss']:.4f}", f"{r_ax['pure_loss']:.4f}",
                f"{r_b['pure_loss']:.4f}"]
        fw.writerow(row)
        rows.append(row)
        if (i + 1) % 20 == 0:
            fh.flush()
            el = time.time() - t0
            print(f"  {i+1}/{len(samples)}  {el:.0f}s "
                  f"(ETA {el/(i+1)*(len(samples)-i-1):.0f}s)", flush=True)
    fh.close()

    # ---------------- 终版汇总 ----------------
    arr = np.array([[float(x) for x in r[2:2 + 20]] for r in rows])
    names = ["nn", "m1", "m1cd", "m2cd", "m2cdx"]
    cols = {n: arr[:, i * 4:(i + 1) * 4] for i, n in enumerate(names)}
    PIX_MM = 0.344

    def poni_px(e):
        return (e[:, 1] + e[:, 2]) / 2.0 / PIX_MM

    print(f"\n=== 全量 {len(rows)} 张（{time.time()-t0:.0f}s）===")
    print(f"{'配方':7s}{'distMAE':>9s}{'dist中位':>9s}{'<0.5mm':>8s}"
          f"{'poni中位px':>11s}{'poni_p75':>9s}{'poni_p90':>9s}"
          f"{'rot2中位':>9s}{'rot2_p90':>9s}{'>5mm':>5s}")
    for n in names:
        e = cols[n]
        pp = poni_px(e)
        print(f"{n:7s}{e[:,0].mean():9.3f}{np.median(e[:,0]):9.3f}"
              f"{(e[:,0]<0.5).mean()*100:7.1f}%{np.median(pp):11.2f}"
              f"{np.percentile(pp,75):9.2f}{np.percentile(pp,90):9.2f}"
              f"{np.median(e[:,3]):9.3f}{np.percentile(e[:,3],90):9.3f}"
              f"{(e[:,0]>5).sum():5d}")

    def sign(w, n):
        try:
            from scipy.stats import binomtest
            return float(binomtest(w, n, 0.5).pvalue)
        except Exception:
            return float("nan")

    print()
    for base in ("m1", "m1cd", "m2cd"):
        for cand in ("m1cd", "m2cd", "m2cdx"):
            if base == cand:
                continue
            a, b = cols[base], cols[cand]
            for col, cn in ((0, "dist"), (3, "rot2")):
                d = b[:, col] - a[:, col]
                w = int((d < 0).sum())
                print(f"符号检验 {base}→{cand} ({cn})：胜 {w}/{len(d)}  p={sign(w, len(d)):.3e}")
    # 采纳率
    ad = np.array([[int(rows[i][22]), int(rows[i][23]), int(rows[i][24])]
                   for i in range(len(rows))])
    print(f"\n采纳率：m1cd {ad[:,0].sum()}/{len(rows)}  "
          f"m2cd {ad[:,1].sum()}/{len(rows)}  m2cdx {ad[:,2].sum()}/{len(rows)}")
    print(f"落盘 {csv_fp}")


if __name__ == "__main__":
    main()
