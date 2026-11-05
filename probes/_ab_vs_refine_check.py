# -*- coding: utf-8 -*-
"""对照：A/B 落盘的逐样本误差 vs tier_eval refine 的逐样本误差（train_v2/practice）。"""
import csv
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "probes", "results")

npz_fp = sorted(glob.glob(os.path.join(RES, "map_prior_ab_*.npz")))[-1]
z = np.load(npz_fp)
key = [k for k in z.files if k.startswith("实际工况档") and "C0" in k][0]
e = z[key]                      # (30, 6) 单位 mm / mrad
poni_px = (e[:, 1] + e[:, 2]) / 2 / 0.344
print(f"A/B  {key}  N={len(e)}")
print(f"  poni px: 中位 {np.median(poni_px):.2f}  p25 {np.percentile(poni_px,25):.2f} "
      f"p75 {np.percentile(poni_px,75):.2f}  max {poni_px.max():.2f}")
print(f"  dist mm: 中位 {np.median(e[:,0]):.3f}  max {e[:,0].max():.3f}")
print(f"  逐样本 poni px: {np.round(np.sort(poni_px),2)}")

csv_fp = sorted(glob.glob(os.path.join(RES, "tier_eval_refine_*.csv")))[-1]
rows = [r for r in csv.DictReader(open(csv_fp, encoding="utf-8"))
        if r.get("dist_err") and r["dataset"] == "train_v2"
        and r["model"] == "base_v2_best" and r["tier"] == "practice"]
d = np.array([float(r["dist_err"]) for r in rows])
p1 = np.array([float(r["poni1_err"]) for r in rows])
p2 = np.array([float(r["poni2_err"]) for r in rows])
poni2_px = (p1 + p2) / 2 / 0.344
print(f"\ntier_eval train_v2/base_v2/practice  N={len(rows)}")
print(f"  poni px: 中位 {np.median(poni2_px):.2f}  p25 {np.percentile(poni2_px,25):.2f} "
      f"p75 {np.percentile(poni2_px,75):.2f}  max {poni2_px.max():.2f}")
print(f"  dist mm: 中位 {np.median(d):.3f}  max {d.max():.3f}")
print(f"  逐样本 poni px: {np.round(np.sort(poni2_px),2)}")
