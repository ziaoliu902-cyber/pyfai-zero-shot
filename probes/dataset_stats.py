# -*- coding: utf-8 -*-
"""全量数据集统计报告：读全部 196 分片的 params/aux（不读 imgs，快），
输出文本摘要 + preview/dataset_stats.png。训练前最后体检。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "train_v1")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preview")

with open(os.path.join(ROOT, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
cfg = meta["config"]

shards = sorted(f for f in os.listdir(ROOT) if f.endswith(".npz"))
params_l, aux_l = [], []
for fn in shards:
    with np.load(os.path.join(ROOT, fn)) as z:
        params_l.append(z["params"])
        aux_l.append(z["aux"])
P = np.concatenate(params_l)      # (N,6) dist,poni1,poni2,rot1,rot2,rot3
A = np.concatenate(aux_l)         # (N,5) phase_id,n_vis,qmax,scale,bs_r
N = len(P)
print(f"N = {N}  shards = {len(shards)}")

names = meta["phase_names"]
phase_counts = np.bincount(A[:, 0].astype(int), minlength=len(names))
print("phase:", {n: int(c) for n, c in zip(names, phase_counts)})

# ---- 几何参数分布 ----
Hm = 521 * cfg["binning"] * 172e-6 * 0.5   # 不用——直接用像素分数即可
lg = np.log10(P[:, 0])
print(f"dist  log10 范围 [{lg.min():.3f},{lg.max():.3f}]  "
      f"十份位计数 {np.histogram(lg, bins=10)[0]}")
for i, nm in enumerate(["poni1", "poni2"]):
    ext = (521 if i == 0 else 490) * cfg["binning"] * 172e-6  # Pilatus 172um native
    frac = P[:, i+1] / (ext / cfg["binning"] * cfg["binning"]) # 分数按 binned extent
print("poni 分数范围:",
      f"poni1 [{P[:,1].min()/ (521*344e-6):.3f},{P[:,1].max()/(521*344e-6):.3f}]",
      f"poni2 [{P[:,2].min()/(490*344e-6):.3f},{P[:,2].max()/(490*344e-6):.3f}]")
for i, nm in enumerate(["rot1", "rot2", "rot3"]):
    r = P[:, i+3]
    lo, hi = cfg[f"{nm}_range"]
    print(f"{nm}: [{r.min():+.4f},{r.max():+.4f}] rad  采样域[{lo:+},{hi:+}]  "
          f"域外比例 {np.mean((r<lo)|(r>hi)):.4f}")

# ---- aux 体检 ----
nvis = A[:, 1]
print(f"n_vis: min={nvis.min():.0f} mean={nvis.mean():.2f}  <2 比例={np.mean(nvis<2):.4f}")
bs = A[:, 4]
print(f"挡光盘: 有 {np.mean(bs>0)*100:.1f}%  半径范围 [{bs[bs>0].min():.0f},{bs.max():.0f}] px")
print(f"qmax: [{A[:,2].min():.2f},{A[:,2].max():.2f}]  scale: "
      f"[{A[:,3].min():.3g},{A[:,3].max():.3g}]")

# ---- 画图 ----
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
axes[0,0].bar(names, phase_counts, color="#4C72B0"); axes[0,0].set_title("phase")
axes[0,1].hist(lg, bins=50, color="#55A868"); axes[0,1].set_title("log10(dist/m)")
axes[0,2].hist(P[:,0], bins=50, color="#55A868"); axes[0,2].set_title("dist/m")
axes[1,0].hist(P[:,1]/(521*344e-6), bins=50); axes[1,0].set_title("poni1 frac")
axes[1,1].hist(P[:,2]/(490*344e-6), bins=50); axes[1,1].set_title("poni2 frac")
axes[1,2].hist(np.degrees(P[:,3]), bins=50); axes[1,2].set_title("rot1 deg")
axes[2,0].hist(np.degrees(P[:,4]), bins=50); axes[2,0].set_title("rot2 deg")
axes[2,1].hist(np.degrees(P[:,5]), bins=50); axes[2,1].set_title("rot3 deg")
axes[2,2].hist(nvis, bins=np.arange(0.5, 12.5), color="#C44E52"); axes[2,2].set_title("visible rings")
for ax in axes.ravel():
    ax.tick_params(labelsize=8)
fig.suptitle(f"train_v1 dataset stats  N={N}", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
fp = os.path.join(OUT, "dataset_stats.png")
fig.savefig(fp, dpi=110)
print("saved", fp)
