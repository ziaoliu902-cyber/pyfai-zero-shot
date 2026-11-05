# -*- coding: utf-8 -*-
"""汇总 tier_eval refine 的逐样本 CSV → 分档对比表。"""
import csv
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fp = sorted(glob.glob(os.path.join(ROOT, "probes", "results", "tier_eval_refine_*.csv")))[-1]
rows = list(csv.DictReader(open(fp, encoding="utf-8")))

groups = {}
for r in rows:
    if not r.get("dist_err"):
        continue
    k = (r["dataset"], r["model"], r["tier"])
    groups.setdefault(k, []).append(r)

print(f"源文件 {fp}\n")
hdr = (f"{'数据集':10s}{'模型':16s}{'档':10s}{'N':>4s}{'dist中位':>10s}{'dist MAE':>10s}"
       f"{'<0.5mm':>8s}{'poni中位px':>11s}{'rot2中位':>9s}{'卡住%':>7s}")
print(hdr)
print("-" * len(hdr))
for k in sorted(groups):
    g = groups[k]
    d = np.array([float(r["dist_err"]) for r in g])
    p1 = np.array([float(r["poni1_err"]) for r in g])
    p2 = np.array([float(r["poni2_err"]) for r in g])
    r2 = np.array([float(r["rot2_err"]) for r in g])
    poni_mm = (p1 + p2) / 2
    poni_px = poni_mm / 0.344  # 像素 344 μm
    print(f"{k[0]:10s}{k[1]:16s}{k[2]:10s}{len(g):4d}{np.median(d):10.3f}"
          f"{d.mean():10.3f}{(d < 0.5).mean()*100:7.0f}%{np.median(poni_px):11.2f}"
          f"{np.median(r2):9.3f}{(d > 5).mean()*100:6.0f}%")
