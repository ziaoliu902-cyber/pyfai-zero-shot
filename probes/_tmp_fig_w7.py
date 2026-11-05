# -*- coding: utf-8 -*-
"""_tmp_fig_w7.py：W7 敏感性双面板图（论文 Fig 素材，英文标签）。"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
OUT = os.path.join(ROOT, "probes", "results", "fig_sensitivity_w7.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

# Panel A：传播系数
labels = ["λ error", "lattice Q error", "pixel size error"]
vals = [-1.397, -1.397, +1.000]
colors = ["#c0504d", "#c0504d", "#4f81bd"]
bars = ax1.bar(labels, vals, color=colors, width=0.55)
ax1.axhline(0, color="k", lw=0.8)
ax1.axhline(-1, color="gray", ls="--", lw=0.8,
            label="small-angle limit (±1)")
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + (0.06 if v > 0 else -0.10),
             f"{v:+.3f}", ha="center", fontsize=9)
ax1.set_ylabel(r"$d\ln D \,/\, d\ln(\mathrm{prior})$")
ax1.set_title("(a) Prior-error propagation\n(CeO$_2$, seven distances)",
              fontsize=10)
ax1.legend(fontsize=8, loc="lower left")
ax1.set_ylim(-1.8, 1.5)
ax1.tick_params(axis="x", labelsize=8)

# Panel B：噪声敏感性
alpha = np.array([5.0, 2.0, 1.0, 0.5, 0.2, 0.05])
med = np.array([0.10, 0.10, 0.13, 0.20, 0.11, 0.23])
worst = np.array([0.29, 0.29, 0.29, 0.29, 0.27, 0.60])
ax2.plot(alpha, med, "o-", color="#4f81bd", label="median |ΔD|")
ax2.plot(alpha, worst, "s--", color="#c0504d", label="worst |ΔD|")
ax2.set_xscale("log")
ax2.set_xlabel("relative exposure (photon statistics)")
ax2.set_ylabel("|ΔD| (mm)")
ax2.set_title("(b) Noise sensitivity\n(Poisson resampling, 7/7 success "
              "at all levels)", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("saved:", OUT)
