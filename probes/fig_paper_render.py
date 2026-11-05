# -*- coding: utf-8 -*-
"""W8 论文图渲染：Fig 2（示例帧+拟合环）、Fig 3（ceria 七档误差）、
Fig 4（消融+归因）、Fig 6（2θ 全几何残差）。
数据源：docs 11/12 号 + 论文初稿 §4 表；真实图 data/real。
输出：probes/results/fig{2,3,4,6}_*.png
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import fabio
import pyFAI.geometry
import pyFAI.detectors

from torchfai.phases import phase_qs

RES = os.path.join(ROOT, "probes", "results")
REAL = os.path.join(ROOT, "data", "real")


def load_geom_with_shape(poni_path, shape):
    g = pyFAI.geometry.Geometry()
    g.load(poni_path)
    det = pyFAI.detectors.Detector(pixel1=g.pixel1, pixel2=g.pixel2)
    det.max_shape = tuple(shape)
    det.shape = tuple(shape)
    g.detector = det
    return g


def block_mean(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def panel_frame_rings(ax, img, poni_path, lam_m, phase, title, f=4,
                      cmap="magma"):
    g = load_geom_with_shape(poni_path, img.shape)
    tth = g.array_from_unit(shape=img.shape, typ="center",
                            unit="2th_deg")
    qs = phase_qs(phase)
    two_th = np.degrees(2 * np.arcsin(qs * lam_m * 1e10 / (4 * np.pi)))
    v = np.log1p(np.clip(img.astype(np.float32), 0, None))
    vd, td = block_mean(v, f), block_mean(tth, f)
    ax.imshow(vd, cmap=cmap, origin="lower",
              extent=[0, img.shape[1], 0, img.shape[0]])
    levels = [t for t in two_th if np.nanmin(td) < t < np.nanmax(td)]
    if levels:
        ax.contour(td, levels=sorted(levels), colors="cyan",
                   linewidths=0.55, alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (px)", fontsize=8)
    ax.set_ylabel("y (px)", fontsize=8)
    ax.tick_params(labelsize=7)


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    # (a) ceria 300 mm
    img = fabio.open(os.path.join(
        REAL, "MX_ceria", "ceria_300_1_0001.cbf")).data
    panel_frame_rings(
        axes[0], img, os.path.join(RES, "ceria_300.poni"),
        9.72386e-11, "CeO2",
        "(a) CeO$_2$ @ 300 mm — fitted rings (classic path)", f=4)
    # (b) alumina mean frame
    files = sorted(x for x in os.listdir(os.path.join(REAL, "alumina"))
                   if x.endswith(".edf"))
    acc = None
    for fn in files:
        d = fabio.open(os.path.join(REAL, "alumina", fn)).data
        acc = d.astype(np.float64) if acc is None else acc + d
    img_mean = (acc / len(files)).astype(np.float32)
    panel_frame_rings(
        axes[1], img_mean, os.path.join(RES, "alumina_mean.poni"),
        7.084811024e-11, "Al2O3",
        f"(b) Al$_2$O$_3$ mean of {len(files)} frames — fitted rings", f=4)
    fig.tight_layout()
    out = os.path.join(RES, "fig2_frames_rings.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


def fig3():
    frames = [150, 200, 250, 300, 350, 400, 450]
    dD = np.array([-0.1, -0.1, -0.0, -0.0, 0.1, 0.1, -0.1])
    rms = np.array([0.58, 0.81, 0.74, 0.65, 0.13, 0.38, 0.29])
    rings = [8, 8, 8, 6, 5, 5, 3]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = ["#2b6cb0" if r >= 5 else "#63b3ed" for r in rings]
    ax.bar([str(f) for f in frames], dD, color=colors, width=0.62,
           edgecolor="k", linewidth=0.4)
    for i, (v, r) in enumerate(zip(dD, rings)):
        ax.text(i, v + (0.02 if v >= 0 else -0.05), f"m={r}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color="#333")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("nominal distance (mm)")
    ax.set_ylabel(r"$\Delta D$ fit (mm)")
    ax.set_ylim(-0.45, 0.3)
    ax.set_title("CeO$_2$ seven distances: mean |ΔD| = 0.1 mm, "
                 "worst 0.1 mm (native 2D refinement)\n"
                 "dark bars: ≥5 rings; light: <5 rings (450 mm still solved "
                 "with 3)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RES, "fig3_ceria_errors.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


def fig4():
    bins = ["0–0.05", "0.05–0.15", "0.15–0.45", ">0.45"]
    n = [3, 6, 30, 46]
    arms = {
        "A NN direct": [0, 17, 3, 0],
        "B geo only": [0, 0, 0, 0],
        "C NN seed": [67, 33, 7, 0],
        "E cascade geo/RS/ray": [67, 33, 30, 2],
        "F +NN (m-first)": [100, 67, 27, 2],
        "G +NN (consensus)": [100, 50, 27, 0],
    }
    fig, (ax, axd) = plt.subplots(
        1, 2, figsize=(10.5, 4.0),
        gridspec_kw={"width_ratios": [2.1, 1]})
    x = np.arange(len(bins))
    w = 0.13
    for k, (name, vals) in enumerate(arms.items()):
        ax.bar(x + (k - 2.5) * w, vals, w, label=name)
    for i, ni in enumerate(n):
        ax.text(i, 104, f"n={ni}", ha="center", fontsize=8, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("beam-offset / canvas diagonal")
    ax.set_ylabel("success rate |ΔD|/D < 1%  (%)")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=7.5, ncol=2)
    ax.set_title("Initializer ablation, synthetic off-axis (85 frames)",
                 fontsize=10)
    # attribution donut
    attr = [("RS", 10), ("NN", 5), ("ray", 1)]
    vals = [a[1] for a in attr]
    axd.pie(vals, labels=[f"{k}\n({v}/16)" for k, v in attr],
            colors=["#2b6cb0", "#dd6b20", "#38a169"],
            wedgeprops=dict(width=0.42, edgecolor="w"), startangle=90)
    axd.text(0, 0, "16 wins\n(geo 0)", ha="center", va="center",
             fontsize=10)
    axd.set_title("Winner attribution", fontsize=10)
    fig.tight_layout()
    out = os.path.join(RES, "fig4_ablation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


def fig6():
    labels = ["150", "200", "250", "300", "350", "400", "450",
              "Al$_2$O$_3$\nmean"]
    mean = [0.92, -0.98, 0.12, 0.21, -0.56, -1.52, 0.02, 1.12]
    rms = [2.41, 1.27, 1.50, 1.66, 1.36, 1.64, 1.30, 1.65]
    mx = [7.75, 2.85, 3.56, 3.27, 2.98, 3.02, 2.87, 3.60]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x, rms, 0.6, color="#2c7a7b", edgecolor="k", linewidth=0.4,
           label="rms")
    ax.scatter(x, mx, marker="_", s=700, color="#c53030",
               linewidths=1.8, label="max |Δ|")
    ax.scatter(x, mean, marker="o", s=18, color="#1a202c",
               label="mean")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("CeO$_2$ frame (mm) / Al$_2$O$_3$ mean")
    ax.set_ylabel("full-geometry 2θ residual (mrad)")
    ax.set_title("2θ residuals vs reference PONIs over the full detector"
                 "\nrms 1.3–2.4 mrad; max ≤ 7.8 mrad", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RES, "fig6_tth_residuals.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "2346"
    if "2" in which:
        fig2()
    if "3" in which:
        fig3()
    if "4" in which:
        fig4()
    if "6" in which:
        fig6()
    print("done")

