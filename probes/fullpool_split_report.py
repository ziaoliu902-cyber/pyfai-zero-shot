# -*- coding: utf-8 -*-
"""把 fullpool_split CSV（多配方）出成论文级终版表。

用法：python probes/fullpool_split_report.py [csv路径]
"""
import csv
import io
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "probes", "results")
PIX_MM = 0.344

RECIPES = [("nn", "NN 粗解（无精修）"),
           ("m1", "M1  refine(σ 笼 + σ 先验)"),
           ("m1cd", "M1cd refine_dual(σ·T 笼 + σ·T 先验)"),
           ("m2cd", "M2cd refine_dual(σ 笼 + σ·T 先验)  ← 解耦"),
           ("m2cdx", "M2cdx refine_dual(1.5σ 笼 + σ·T 先验)")]
CH = ("dist", "poni1", "poni2", "rot2")


def sign_p(w, n):
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import binomtest
        return float(binomtest(w, n, 0.5).pvalue)
    except Exception:
        from math import comb
        k = min(w, n - w)
        return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def wilcoxon_p(d):
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        cands = sorted(f for f in os.listdir(OUTDIR)
                       if f.startswith("fullpool_split_") and f.endswith(".csv"))
        if not cands:
            print("找不到 fullpool_split CSV")
            return 1
        path = os.path.join(OUTDIR, cands[-1])
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    ph = np.array([r["phase"] for r in rows])

    def col(rec, k):
        return np.array([float(r[f"{rec}_{k}"]) for r in rows])

    E = {}
    for rec, _ in RECIPES:
        E[rec] = np.stack([col(rec, k) for k in CH], axis=1)

    def poni_px(e):
        return (e[:, 1] + e[:, 2]) / 2.0 / PIX_MM

    out = []
    W = out.append
    W("# 实际工况档全池：σ 用在信任域还是软先验？（全量 A/B 终版）\n")
    W(f"- 数据：`data/train_v3` test，实际工况档全池 **N={n}**")
    W(f"- 粗解：`runs/base_v3/last.pt`；iters=400, trust_k=1.0, soft_prior=0.1, adopt_nats=1.0")
    W(f"- 像素 {PIX_MM} mm/px（poni 误差 = (|Δponi1|+|Δponi2|)/2 / px）")
    W(f"- 原始数据：`{os.path.relpath(path, ROOT)}`\n")

    W("## 1. 主表\n")
    W("| 配方 | dist MAE | dist 中位 | <0.5mm | >5mm | poni 中位px | poni p75 | poni p90 | rot2 中位 | rot2 p90 |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for rec, label in RECIPES:
        e = E[rec]
        pp = poni_px(e)
        s = dict(d_mae=e[:, 0].mean(), d_med=np.median(e[:, 0]),
                 d_p75=np.percentile(e[:, 0], 75), d_p90=np.percentile(e[:, 0], 90),
                 lt05=(e[:, 0] < 0.5).mean() * 100, gt5=int((e[:, 0] > 5).sum()),
                 p_med=np.median(pp), p_p75=np.percentile(pp, 75),
                 p_p90=np.percentile(pp, 90), r_med=np.median(e[:, 3]),
                 r_p90=np.percentile(e[:, 3], 90), r_gt20=int((e[:, 3] > 20).sum()))
        stats[rec] = s
        W(f"| {label} | {s['d_mae']:.3f} | {s['d_med']:.3f} | {s['lt05']:.1f}% | "
          f"{s['gt5']} | {s['p_med']:.2f} | {s['p_p75']:.2f} | {s['p_p90']:.2f} | "
          f"{s['r_med']:.3f} | {s['r_p90']:.3f} |")
    W("")

    W("## 2. 配方两两配对检验\n")
    W("| 对比 | 通道 | 优 | 劣 | 平 | 符号检验 p | Wilcoxon p |")
    W("|---|---|---|---|---|---|---|")
    pairs = []
    for i in range(len(RECIPES)):
        for j in range(i + 1, len(RECIPES)):
            a, b = RECIPES[i][0], RECIPES[j][0]
            if a == "nn":
                continue
            for ci, cn in ((0, "dist"), (3, "rot2")):
                d = E[b][:, ci] - E[a][:, ci]
                w = int((d < 0).sum())
                lo = int((d > 0).sum())
                z = int((d == 0).sum())
                p = sign_p(w, n)
                pw = wilcoxon_p(d)
                W(f"| {a} → {b} | {cn} | {w} | {lo} | {z} | {p:.3e} | {pw:.3e} |")
                pairs.append(dict(a=a, b=b, ch=cn, win=w, lose=lo, tie=z,
                                  p=p, wilcoxon_p=pw))
    W("")

    W("## 3. 双轨采纳拆解\n")
    for rec in ("m1cd", "m2cd", "m2cdx"):
        col = f"adopt_{rec}"
        if col not in rows[0]:
            continue
        ad = np.array([int(r[col]) for r in rows])
        W(f"### {rec}：采纳 {int(ad.sum())}/{n} = {ad.mean()*100:.1f}%")
        for tag, msk in (("采纳B轨", ad == 1), ("保留A轨", ad == 0)):
            if msk.sum() == 0:
                continue
            W(f"- **{tag}**（{int(msk.sum())} 张）：dist 中位 "
              f"{np.median(E[rec][msk][:,0]):.3f} mm，poni 中位 "
              f"{np.median(poni_px(E[rec][msk])):.2f} px")
        W("")

    W("## 4. 分相拆解（dist 中位 mm / poni 中位 px / rot2 中位 mrad）\n")
    W("| 相 | N | " + " | ".join(f"{r}" for r, _ in RECIPES) + " |")
    W("|---|---|" + "---|" * len(RECIPES))
    for p_ in sorted(set(ph)):
        m = ph == p_
        cells = []
        for rec, _ in RECIPES:
            e = E[rec][m]
            cells.append(f"{np.median(e[:,0]):.3f} / {np.median(poni_px(e)):.2f} / {np.median(e[:,3]):.3f}")
        W(f"| {p_} | {int(m.sum())} | " + " | ".join(cells) + " |")
    W("")

    W("## 5. 尾部与误伤\n")
    for rec, _ in RECIPES:
        if rec == "nn":
            continue
        e = E[rec]
        W(f"- **{rec}**：dist p50/p75/p90/p95 = "
          f"{np.percentile(e[:,0],50):.3f} / {np.percentile(e[:,0],75):.3f} / "
          f"{np.percentile(e[:,0],90):.3f} / {np.percentile(e[:,0],95):.3f} mm；"
          f">2mm {int((e[:,0]>2).sum())} 张；>5mm {int((e[:,0]>5).sum())} 张；"
          f">10mm {int((e[:,0]>10).sum())} 张")
    W("")
    base = "m1"
    for rec in ("m1cd", "m2cd", "m2cdx"):
        d = E[rec][:, 0] - E[base][:, 0]
        W(f"- {rec} vs {base}：变差>1mm {int((d>1).sum())} 张，"
          f"变差>0.2mm {int((d>0.2).sum())} 张，"
          f"变好>1mm {int((d<-1).sum())} 张，"
          f"均值 {d.mean():+.4f} mm，中位 {np.median(d):+.4f} mm")
    W("")

    ts = time.strftime("%Y%m%d_%H%M%S")
    md_fp = os.path.join(OUTDIR, f"fullpool_split_report_{ts}.md")
    io.open(md_fp, "w", encoding="utf-8").write("\n".join(out))
    js_fp = os.path.join(OUTDIR, f"fullpool_split_report_{ts}.json")
    io.open(js_fp, "w", encoding="utf-8").write(json.dumps(
        dict(n=n, stats={k: {kk: float(vv) for kk, vv in v.items()}
                         for k, v in stats.items()},
             pairs=[{k: (float(v) if isinstance(v, float) else v)
                     for k, v in p.items()} for p in pairs],
             csv=os.path.relpath(path, ROOT)), ensure_ascii=False, indent=2))
    print("\n".join(out))
    print(f"\n落盘 {md_fp}\n落盘 {js_fp}")


if __name__ == "__main__":
    sys.exit(main() or 0)
