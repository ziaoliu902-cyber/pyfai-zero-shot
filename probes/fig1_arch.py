# -*- coding: utf-8 -*-
"""Fig 1 架构图：classic-first 级联 + 跨种子共识门。
纯 matplotlib 矢量风格框图，输出 probes/results/fig1_architecture.png
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE = "#2b6cb0"
LBLUE = "#ebf4ff"
GREEN = "#2f855a"
LGREEN = "#e6f4ea"
ORANGE = "#c05621"
LORANGE = "#fdf0e7"
GRAY = "#4a5568"
LGRAY = "#f0f0f0"


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, bold=False, sub=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,"
                       "rounding_size=0.02", fc=fc, ec=ec, lw=1.3)
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.63, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color="#1a202c")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=fs - 1.8, color=GRAY, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color="#1a202c")
    return (x, y, w, h)


def arrow(ax, p1, p2, color=GRAY, label=None, lx=0, ly=0, lw=1.4,
          dashed=False, fs=7.5):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                        color=color, lw=lw,
                        linestyle="--" if dashed else "-",
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
        ax.text(mx, my, label, fontsize=fs, color=color, ha="center",
                backgroundcolor="white")


def main():
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # ── 输入 ──
    box(ax, 0.15, 6.35, 1.9, 0.95, "Diffraction\nframe", LGRAY, GRAY,
        bold=True, sub="calibrant rings")
    box(ax, 0.15, 4.9, 1.9, 0.95, "Phase library\n+ priors", LGRAY, GRAY,
        sub="λ, d(hkl), px")

    # ── 主路径（classic-first，15 ms）──
    ax.text(5.35, 7.35, "production path — 15 ms/frame, no model loaded",
            ha="center", fontsize=9, color=GREEN, fontweight="bold")
    b_canvas = box(ax, 2.7, 6.35, 1.85, 0.95, "Canvas\nrebin", LGREEN,
                   GREEN, sub="keep beam offset")
    b_geo = box(ax, 5.2, 6.35, 2.05, 0.95, "Classic solver\n(geo center)",
                LGREEN, GREEN, sub="vote + pair + LSQ")
    b_out = box(ax, 8.0, 6.35, 1.8, 0.95, "PONI +\nuncertainty", LGREEN,
                GREEN, bold=True, sub="2θ check")
    arrow(ax, (2.05, 6.82), (2.7, 6.82))
    arrow(ax, (2.05, 5.37), (2.85, 4.6), label="priors", lx=0.35, ly=0.25)
    arrow(ax, (4.55, 6.82), (5.2, 6.82))
    arrow(ax, (7.25, 6.82), (8.0, 6.82))

    # 通过门
    arrow(ax, (6.2, 6.35), (6.2, 5.7), color=GREEN,
          label=r"$\geq$3 rings matched", lx=1.25, ly=0.05)
    arrow(ax, (8.9, 6.35), (8.9, 5.7), color=GREEN)
    box(ax, 8.05, 4.85, 1.75, 0.85, "tilt refine\n+ poni write", LGREEN,
        GREEN, fs=8)
    arrow(ax, (8.9, 4.85), (8.9, 6.35), color=GREEN)  # placeholder no

    # ── 兜底（fallback bank）──
    ax.text(4.1, 4.35, "fallback bank (classic failure only)",
            ha="center", fontsize=9, color=ORANGE, fontweight="bold")
    arrow(ax, (6.2, 5.7), (6.2, 3.95), color=ORANGE, dashed=True,
          label="fail", lx=0.55, ly=-0.1)

    box(ax, 0.5, 2.15, 2.1, 1.15, "RS center\n(closed form)", LORANGE,
        ORANGE, sub="grad-line LSQ")
    box(ax, 2.95, 2.15, 2.1, 1.15, "Ray seeds ×3\n(extrapolate)", LORANGE,
        ORANGE, sub="fix shrinkage")
    box(ax, 5.4, 2.15, 2.1, 1.15, "NN prior\n(v4s1)", LORANGE, ORANGE,
        sub="±80 mm basin")
    box(ax, 7.85, 2.15, 2.0, 1.15, "Geo center\n(canvas)", LORANGE,
        ORANGE, sub="re-seed")

    for x0 in (1.55, 4.0, 6.45, 8.85):
        arrow(ax, (x0, 3.3), (x0 if x0 > 4 else 4.6, 1.75),
              color=ORANGE, dashed=True)

    # ── 共识门 ──
    box(ax, 2.6, 0.7, 4.4, 1.05,
        "Cross-seed consensus gate", "#fff5f5", "#c53030", bold=True,
        sub="votes → pscore → rings → rms")
    arrow(ax, (7.0, 1.22), (8.9, 2.15), color="#c53030",
          label="best", lx=0.55, ly=0.3)

    # NN 训练支线
    box(ax, 0.15, 0.7, 1.9, 1.05, "synthetic\ntraining 10⁵", LBLUE, BLUE,
        fs=8, sub="6 phases")
    arrow(ax, (2.05, 1.22), (2.6, 1.22), color=BLUE, dashed=True,
          label="offline", ly=0.24)

    ax.text(0.12, 0.12,
            "True basin is re-discovered by independent seeds; "
            "pseudo-solutions stay singletons →\nconsensus removes "
            "catastrophic picks (ceria: 3/7 → 0/7) without touching the "
            "15 ms path.",
            fontsize=8, color=GRAY, style="italic")

    out = os.path.join(ROOT, "probes", "results",
                       "fig1_architecture.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
