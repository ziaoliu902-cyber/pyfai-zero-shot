# -*- coding: utf-8 -*-
"""从 fullpool_dual CSV 出「实际工况档」论文终版数字表。

用法：python probes/fullpool_report.py [csv路径]

输出（stdout + probes/results/fullpool_report_<ts>.md）：
  - M1 vs M1cd 全表：dist MAE/中位/p75/p90/<0.5mm/>5mm、
    poni 中位/p75/p90(px)、rot2 中位/p90
  - 配对符号检验（dist / poni / rot2）+ Wilcoxon 符号秩（若有 scipy）
  - 分相（phase）拆解
  - 误伤清单：M1cd 比 M1 明显变差的样本
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
PIX_MM = 0.000344 * 1000.0          # Pilatus_1M binning=2 → 0.344 mm/px

CSV_DEFAULT = os.path.join(OUTDIR, "fullpool_dual_20261027_112228.csv")


def sign_test(w: int, n: int) -> float:
    """双侧配对符号检验 p 值（零差异按 0.5 计入）。"""
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import binomtest
        return float(binomtest(w, n, 0.5).pvalue)
    except Exception:
        # 精确二项分布（无 scipy）
        from math import comb
        k = min(w, n - w)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        return min(1.0, 2.0 * tail)


def wilcoxon(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    if not os.path.isfile(path):
        cand = sorted([os.path.join(OUTDIR, f) for f in os.listdir(OUTDIR)
                       if f.startswith("fullpool_dual_") and f.endswith(".csv")])
        if not cand:
            print(f"找不到 fullpool_dual CSV：{path}")
            return 1
        path = cand[-1]
    rows = load(path)
    n = len(rows)
    ph = np.array([r["phase"] for r in rows])
    adopt = np.array([int(r["adopt"]) for r in rows])
    m1 = np.array([[float(r[f"m1_{k}"]) for k in
                    ("dist", "poni1", "poni2", "rot2")] for r in rows])
    m2 = np.array([[float(r[f"m1cd_{k}"]) for k in
                    ("dist", "poni1", "poni2", "rot2")] for r in rows])

    def poni_px(e):
        return (e[:, 1] + e[:, 2]) / 2.0 / PIX_MM

    out = []
    W = out.append
    W(f"# 实际工况档全池评测终版（M1 vs M1cd）\n")
    W(f"- 数据：`data/train_v3` test split，实际工况档全池 **N={n}**")
    W(f"- 粗解：`runs/base_v3/last.pt`（NN 粗解 → 可微积分器精修）")
    W(f"- M1 = `refine(σ, trust_k=1.0, soft_prior=0.1)`")
    W(f"- M1cd = `refine_dual(σ·T, trust_k=1.0, soft_prior=0.1, adopt_nats=1.0)`")
    W(f"- σ 标定 T = dist 0.317 / poni1 0.532 / poni2 0.515 / rot2 1.459")
    W(f"- 像素 {PIX_MM:.3f} mm/px（poni 误差按 |Δponi1|+|Δponi2|)/2 折 px）")
    W(f"- 原始数据：`{os.path.relpath(path, ROOT)}`\n")

    W("## 1. 主表\n")
    W("| 指标 | M1 | M1cd | 变化 |")
    W("|---|---|---|---|")
    stats = []
    for name, e in (("M1", m1), ("M1cd", m2)):
        pp = poni_px(e)
        stats.append(dict(
            name=name,
            d_mae=e[:, 0].mean(), d_med=np.median(e[:, 0]),
            d_p75=np.percentile(e[:, 0], 75), d_p90=np.percentile(e[:, 0], 90),
            d_lt05=(e[:, 0] < 0.5).mean() * 100,
            d_gt5=(e[:, 0] > 5).sum(),
            p_med=np.median(pp), p_p75=np.percentile(pp, 75),
            p_p90=np.percentile(pp, 90),
            r_med=np.median(e[:, 3]), r_p90=np.percentile(e[:, 3], 90),
        ))
    s1, s2 = stats

    def line(label, a, b, fmt="%.3f", unit=""):
        ch = b - a
        W(f"| {label} | {fmt % a}{unit} | {fmt % b}{unit} | {ch:+.{len(fmt.split('.')[1]) if '.' in fmt else 2}f}{unit} |")

    line("dist MAE (mm)", s1["d_mae"], s2["d_mae"])
    line("dist 中位 (mm)", s1["d_med"], s2["d_med"])
    line("dist p75 (mm)", s1["d_p75"], s2["d_p75"])
    line("dist p90 (mm)", s1["d_p90"], s2["d_p90"])
    line("dist <0.5mm 占比 (%)", s1["d_lt05"], s2["d_lt05"], "%.1f", "%")
    line("dist >5mm 张数", s1["d_gt5"], s2["d_gt5"], "%.0f", " 张")
    line("poni 中位 (px)", s1["p_med"], s2["p_med"], "%.2f")
    line("poni p75 (px)", s1["p_p75"], s2["p_p75"], "%.2f")
    line("poni p90 (px)", s1["p_p90"], s2["p_p90"], "%.2f")
    line("rot2 中位 (mrad)", s1["r_med"], s2["r_med"])
    line("rot2 p90 (mrad)", s1["r_p90"], s2["r_p90"])
    W("")

    W("## 2. 配对检验（逐样本，同一批图）\n")
    W("| 通道 | M1cd 更优 | M1 更优 | 持平 | 符号检验 p | Wilcoxon p |")
    W("|---|---|---|---|---|---|")
    sig_rows = []
    for col, nm in ((0, "dist (mm)"), (3, "rot2 (mrad)")):
        d = m2[:, col] - m1[:, col]
        w = int((d < 0).sum())
        lo = int((d > 0).sum())
        z = int((d == 0).sum())
        p = sign_test(w, n)
        pw = wilcoxon(d)
        W(f"| {nm} | {w} | {lo} | {z} | {p:.3e} | {pw:.3e} |")
        sig_rows.append((nm, w, lo, z, p, pw))
    dpon = (poni_px(m2) - poni_px(m1))
    w = int((dpon < 0).sum())
    lo = int((dpon > 0).sum())
    z = int((dpon == 0).sum())
    p = sign_test(w, n)
    W(f"| poni (px) | {w} | {lo} | {z} | {p:.3e} | {wilcoxon(dpon):.3e} |")
    sig_rows.append(("poni (px)", w, lo, z, p, wilcoxon(dpon)))
    W("")

    # 采纳子集 vs 未采纳子集
    W("## 3. 双轨采纳行为拆解\n")
    W(f"- 全池采纳率：**{int(adopt.sum())}/{n} = {adopt.mean()*100:.1f}%**")
    for tag, msk in (("采纳B轨", adopt == 1), ("保留A轨", adopt == 0)):
        if msk.sum() == 0:
            continue
        a, b = m1[msk], m2[msk]
        W(f"- **{tag}**（{int(msk.sum())} 张）："
          f"dist 中位 {np.median(a[:,0]):.3f}→{np.median(b[:,0]):.3f} mm，"
          f"poni 中位 {np.median(poni_px(a)):.2f}→{np.median(poni_px(b)):.2f} px，"
          f"rot2 中位 {np.median(a[:,3]):.3f}→{np.median(b[:,3]):.3f} mrad")
    W("")

    # 分相
    W("## 4. 分相拆解\n")
    W("| 相 | N | 采纳率 | M1 dist中位 | M1cd dist中位 | M1 poni中位px | M1cd poni中位px | M1 rot2中位 | M1cd rot2中位 |")
    W("|---|---|---|---|---|---|---|---|---|")
    for p_ in sorted(set(ph)):
        m = ph == p_
        W(f"| {p_} | {int(m.sum())} | {adopt[m].mean()*100:.0f}% | "
          f"{np.median(m1[m][:,0]):.3f} | {np.median(m2[m][:,0]):.3f} | "
          f"{np.median(poni_px(m1[m])):.2f} | {np.median(poni_px(m2[m])):.2f} | "
          f"{np.median(m1[m][:,3]):.3f} | {np.median(m2[m][:,3]):.3f} |")
    W("")

    # 误伤清单
    W("## 5. 误伤清单（M1cd 明显差于 M1 的样本）\n")
    ddist = m2[:, 0] - m1[:, 0]
    bad = np.where(ddist > 1.0)[0]
    worse = np.where((ddist > 0.2))[0]
    W(f"- dist 变差 >1mm：**{len(bad)} 张**")
    W(f"- dist 变差 >0.2mm：{len(worse)} 张")
    W(f"- dist 变好 >1mm：{int((ddist < -1).sum())} 张；"
      f"变好 >0.2mm：{int((ddist < -0.2).sum())} 张")
    W(f"- 逐样本差：均值 {ddist.mean():+.4f} mm，中位 {np.median(ddist):+.4f} mm\n")
    if len(bad):
        W("| # | 相 | M1 dist | M1cd dist | Δ |")
        W("|---|---|---|---|---|")
        for i in bad:
            W(f"| {i} | {ph[i]} | {m1[i,0]:.4f} | {m2[i,0]:.4f} | {ddist[i]:+.4f} |")
        W("")

    # 尾部
    W("## 6. 尾部与鲁棒性\n")
    for q in (50, 75, 90, 95, 99):
        W(f"- dist p{q}: M1 {np.percentile(m1[:,0], q):.3f} mm → "
          f"M1cd {np.percentile(m2[:,0], q):.3f} mm")
    W(f"- dist >2mm: M1 {int((m1[:,0]>2).sum())} 张 → M1cd {int((m2[:,0]>2).sum())} 张")
    W(f"- dist >10mm: M1 {int((m1[:,0]>10).sum())} 张 → M1cd {int((m2[:,0]>10).sum())} 张")
    W(f"- rot2 >20mrad: M1 {int((m1[:,3]>20).sum())} 张 → M1cd {int((m2[:,3]>20).sum())} 张")
    W("")

    ts = time.strftime("%Y%m%d_%H%M%S")
    md_fp = os.path.join(OUTDIR, f"fullpool_report_{ts}.md")
    io.open(md_fp, "w", encoding="utf-8").write("\n".join(out))
    json_fp = os.path.join(OUTDIR, f"fullpool_report_{ts}.json")
    io.open(json_fp, "w", encoding="utf-8").write(json.dumps(
        dict(n=n, adopt=int(adopt.sum()),
             stats={k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                             else int(vv))
                        for kk, vv in v.items() if kk != "name"}
                    for k, v in dict(M1=s1, M1cd=s2).items()},
             sign_tests=[dict(name=a, win=int(b), lose=int(c), tie=int(d),
                              p=float(e), wilcoxon_p=float(f))
                         for a, b, c, d, e, f in sig_rows],
             csv=os.path.relpath(path, ROOT)), ensure_ascii=False, indent=2))
    print("\n".join(out))
    print(f"\n落盘 {md_fp}\n落盘 {json_fp}")


if __name__ == "__main__":
    sys.exit(main() or 0)
