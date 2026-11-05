# -*- coding: utf-8 -*-
"""§4.6 两张主图渲染：Fig 7（条件相图：曝光/倾角/角域/畸变）+
Fig 8（pyFAI duel 对照 + 八件套组件消融）。
数据源：docs/phase_map.md、docs/pyfai_duel.md + probes/results/pyfai_duel_v3.md、
docs/ablation8.md（40 样本合成压力集表）。输出 probes/results/fig7_*.png fig8_*.png
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(ROOT, "probes", "results")

C_T, C_P0, C_P1 = "#2b6cb0", "#c53030", "#dd6b20"   # T / pyFAI P0 / pyFAI P1
C_OK, C_WARN, C_BAD = "#2b6cb0", "#dd6b20", "#c53030"
GRAY = "#666666"


# ----------------------------------------------------------------------
# Fig 7 — condition map（docs/phase_map.md，合成 CeO2，每点 5 seeds）
# ----------------------------------------------------------------------
def fig7():
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))

    # (a) 曝光：ΔD 与 Δcenter 对峰计数（30→30000）双不敏感
    ax = axes[0, 0]
    counts = np.array([30, 100, 300, 1000, 3000, 10000, 30000])
    dD = np.array([0.21, 0.19, 0.25, 0.21, 0.25, 0.49, 0.49])
    dc = np.array([1.41, 1.40, 1.41, 1.41, 1.41, 1.44, 1.44])
    x = np.arange(len(counts))
    ax.bar(x, dD, 0.6, color=C_OK, edgecolor="k", linewidth=0.4,
           label=r"$|\Delta D|$ (mm)")
    for xi, v in zip(x, dD):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5,
                color="#333")
    ax.set_ylim(0, 0.62)
    ax.set_ylabel(r"$|\Delta D|$ (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in counts], fontsize=8)
    ax.set_xlabel("peak counts per ring (Poisson only)")
    ax2 = ax.twinx()
    ax2.plot(x, dc, "o--", color=C_WARN, ms=4, lw=1.1,
             label=r"$\Delta$center (px)")
    ax2.set_ylim(1.30, 1.55)
    ax2.set_ylabel(r"$\Delta$center (px)", color=C_WARN)
    ax2.tick_params(axis="y", labelcolor=C_WARN)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("(a) exposure is not limiting: |ΔD| 0.2–0.5 mm,\n"
                 "Δcenter flat ~1.4 px; phase-ID 100% at all counts",
                 fontsize=9)

    # (b) 倾角：中心误差 ≈23 px/°，与曝光无关
    ax = axes[0, 1]
    tilt = np.array([0.5, 1.0, 2.0, 4.0])
    dcs = np.array([11.68, 22.94, 46.15, 92.08])
    tf = np.linspace(0, 4.3, 50)
    ax.plot(tf, 23.0 * tf, "-", color=GRAY, lw=1.0,
            label="23 px/° fit")
    ax.plot(tilt, dcs, "o", color=C_T, ms=7, zorder=5,
            label="Δcenter (λ known)")
    # 2D 相图三档曝光（100/1000/10000 计数）几乎重合
    for expo, mk in [(100, "x"), (1000, "+"),(10000, "1")]:
        vals = {100: [11.8, 23.1, 46.8, 90.9],
                1000: [11.8, 23.4, 46.8, 90.6],
                10000: [11.7, 22.9, 46.2, 92.4]}[expo]
        ax.plot(tilt, vals, mk, color="#b7791f", ms=4.5, mew=1.1, alpha=0.75,
                label="10$^2$/10$^3$/10$^4$ counts")
        break  # 一个图例项代表三档
    for expo, mk in [(1000, "+"), (10000, "1")]:
        vals = {1000: [11.8, 23.4, 46.8, 90.6],
                10000: [11.7, 22.9, 46.2, 92.4]}[expo]
        ax.plot(tilt, vals, mk, color="#b7791f", ms=4.5, mew=1.1, alpha=0.75)
    for t, v in zip(tilt, dcs):
        ax.text(t + 0.08, v - 3.5, f"{v:.1f}", fontsize=7.5, color=C_T)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("detector tilt rot1 = rot2 (deg)")
    ax.set_ylabel(r"$\Delta$center (px)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(b) tilt is the center killer: Δcenter ≈ 23 px/°,\n"
                 "independent of exposure; |ΔD| stays < 1 mm",
                 fontsize=9)

    # (c) 角域决定 λ–D 可分性：λ̂ 误差 vs D（λ 联合模式）
    ax = axes[1, 0]
    D = np.array([60, 90, 130, 180, 250, 350, 480, 650])
    lam = np.array([0.585, 0.641, 0.494, 0.492, 0.501, 0.506, 0.507, 1.680])
    err = np.abs(lam / 0.5 - 1) * 100
    regime = ["narrow", "narrow", "ok", "ok", "ok", "ok", "ok", "off"]
    cmap_ = {"ok": C_OK, "narrow": C_WARN, "off": C_BAD}
    for rg in ["ok", "narrow", "off"]:
        m = [r == rg for r in regime]
        ax.plot(D[m], err[m], "o-", color=cmap_[rg], ms=6, lw=1.2,
                label={"ok": "in field of view (130–480 mm)",
                       "narrow": "rings squeezed at centre (60–90 mm)",
                       "off": "rings off-canvas (650 mm)"}[rg])
    ax.axhspan(0, 2, color="#2f855a", alpha=0.10)
    ax.axhline(2, ls="--", color="#2f855a", lw=1.0)
    ax.text(640, 2.6, "±2%", color="#2f855a", fontsize=8, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.08, 700)
    ax.set_xlabel(r"sample–detector distance $D$ (mm);  true $\lambda$ = 0.5 Å")
    ax.set_ylabel(r"$|\hat\lambda/\lambda - 1|$  (%)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title(r"(c) $\lambda$ becomes identifiable only with a wide ring"
                 "\n"
                 r"system ($2\theta_{max}\gtrsim30°$); joint solve drifts "
                 "17–28%\nelsewhere",
                 fontsize=9)

    # (d) 真实 pyFAI 样条闭环：clean / raw / corrected
    ax = axes[1, 1]
    cats = ["clean\n(no distortion)", "raw 1% radial\ndistortion",
            "corrected\n(solver undistort)"]
    dcs2 = [1.44, 1.81, 1.43]
    dDs2 = ["−0.50", "+0.39", "−0.18"]
    cols = [C_OK, C_BAD, "#2f855a"]
    bars = ax.bar(cats, dcs2, 0.55, color=cols, edgecolor="k", linewidth=0.4)
    for b, v, ddt in zip(bars, dcs2, dDs2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.055,
                f"{v:.2f} px\nΔD = {ddt} mm", ha="center", fontsize=8)
    ax.axhline(1.44, ls="--", color=GRAY, lw=0.9)
    ax.set_ylim(0, 2.35)
    ax.set_ylabel(r"$\Delta$center (px)")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("(d) real pyFAI spline closed loop: undistort\n"
                 "restores the clean value (1.81 → 1.43 vs 1.44 px)",
                 fontsize=9)

    fig.tight_layout()
    out = os.path.join(RES, "fig7_condition_map.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------------
# Fig 8 — pyFAI duel（v3 全表）+ 组件消融（40 样本压力集）
# ----------------------------------------------------------------------
def fig8():
    frames = ["Pilatus\n1M", "Pilatus\n2M", "Eiger\n4M", "Eiger\n4M_v2",
              "Eiger\n9M", "Frelon\n2k*", "half\nccd*", "mar3450\n(LaB6)",
              "ceria\n150", "ceria\n250", "ceria\n400"]
    n = len(frames)
    spline = {5, 6}   # * = spline 畸变检测器（域外，Limitations v）

    T_dD = np.array([55.80, 0.05, 0.03, 0.10, 0.01, 48.68, 3.87, 5.87,
                     0.06, 0.01, 0.11])
    P0_dD = np.array([583.23, 699.75, 62.56, 61.93, 786.66, 5.74, 0.53,
                      777.45, 49.11, 50.75, 92.28])
    P1_dD = np.array([55.80, 0.05, 0.05, 0.10, 0.01, 693.85, 4.05, 5.87,
                      0.11, 0.20, 0.32])

    lamD_T = np.array([3.52, 0.01, 0.02, 0.06, 0.01, np.nan, np.nan, 0.85,
                       0.04, 0.01, 0.03])
    lamD_P0 = np.array([36.84, 87.48, 38.48, 38.25, 368.97, np.nan, np.nan,
                        349.34, 32.95, 20.36, 23.10])
    lamD_P1 = np.array([3.52, 0.01, 0.03, 0.06, 0.01, np.nan, np.nan, 2.64,
                        0.07, 0.08, 0.08])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

    def shade(ax):
        for i in sorted(spline):
            ax.axvspan(i - 0.5, i + 0.5, color="#cbd5e0", alpha=0.45,
                       zorder=0)

    # (a) |ΔD|（log）三臂对照
    ax = axes[0, 0]
    x = np.arange(n)
    w = 0.27
    shade(ax)
    ax.bar(x - w, T_dD, w, color=C_T, edgecolor="k", linewidth=0.3,
           label="T  ours, zero-shot")
    ax.bar(x, P0_dD, w, color=C_P0, edgecolor="k", linewidth=0.3,
           label="P0 pyFAI, zero-start")
    ax.bar(x + w, P1_dD, w, color=C_P1, edgecolor="k", linewidth=0.3,
           label="P1 pyFAI, started from T")
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 3e3)
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.text(10.45, 1.25, "1 mm", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(frames, fontsize=7)
    ax.set_ylabel(r"$|\Delta D|$ (mm, log)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(a) distance: T sub-mm wherever λ is faithful; P0 spans "
                 "0.5–787 mm\n"
                 "P1M & mar bars sit on the λ–D ridge (panel b); grey band* = "
                 "spline\ndetectors, outside present scope (Limitations v)",
                 fontsize=9)

    # (b) |ΔλD|（log）：简并脊口径
    ax = axes[0, 1]
    shade(ax)
    ax.bar(x - w, lamD_T, w, color=C_T, edgecolor="k", linewidth=0.3,
           label="T")
    ax.bar(x, lamD_P0, w, color=C_P0, edgecolor="k", linewidth=0.3,
           label="P0")
    ax.bar(x + w, lamD_P1, w, color=C_P1, edgecolor="k", linewidth=0.3,
           label="P1")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 2e3)
    ax.axhline(2.0, color="#2f855a", ls="--", lw=1.0)
    ax.text(10.45, 2.6, "2%", fontsize=7.5, color="#2f855a")
    ax.set_xticks(x)
    ax.set_xticklabels(frames, fontsize=7)
    ax.set_ylabel(r"$|\Delta(\hat\lambda D)/\lambda D|$  (%, log)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(b) λD-product metric (geometry fidelity): P0 off by "
                 "20–369%;\n"
                 "T ≤ 0.85% except P1M (absorbs the GT λ offset); "
                 "* frames have no λ\nin the reference PONI → λD undefined",
                 fontsize=9)

    # (c) 组件消融：中位 |ΔD|（40 样本合成压力集）
    ax = axes[1, 0]
    arms = ["full\npipeline", "− auto\ncentering", "− classic\ncascade",
            "− 2D\nrefine", "λ joint\n(narrow span)", "GT phase\n(oracle)"]
    med = [0.92, 4.98, 1.00, 0.89, 28.89, 0.84]
    cols = [C_OK, C_BAD, "#b7791f", "#63b3ed", "#9b2c2c", "#a0aec0"]
    bars = ax.bar(np.arange(6), med, 0.58, color=cols, edgecolor="k",
                  linewidth=0.4)
    for b, v in zip(bars, med):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 80)
    ax.axhline(0.92, ls=":", color=GRAY, lw=0.9)
    ax.text(5.45, 1.05, "full", fontsize=7.5, color=GRAY)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(arms, fontsize=8)
    ax.set_ylabel(r"median $|\Delta D|$ (mm, log)")
    ax.set_title("(c) component ablation (40-frame stress set: D 0.1–0.6 m, "
                 "±3° tilt,\n"
                 "300–10⁴ counts, off-axis centres): auto-centering ×5, "
                 "λ-joint ×31",
                 fontsize=9)

    # (d) 组件消融：相正确率
    ax = axes[1, 1]
    ph = [88, 50, 78, 88, 75, 100]
    bars = ax.bar(np.arange(6), ph, 0.58, color=cols, edgecolor="k",
                  linewidth=0.4)
    for b, v in zip(bars, ph):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v}%",
                ha="center", fontsize=8)
    ax.set_ylim(0, 116)
    ax.axhline(88, ls=":", color=GRAY, lw=0.9)
    ax.axhline(100, ls="--", color="#2f855a", lw=0.9)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(arms, fontsize=8)
    ax.set_ylabel("phase identification (%)")
    ax.set_title("(d) same arms, phase-ID rate: auto-centering 88→50%, "
                 "cascade 88→78%;\noracle 100% bounds the arbitration loss",
                 fontsize=9)

    fig.tight_layout()
    out = os.path.join(RES, "fig8_duel_ablation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "78"
    if "7" in which:
        fig7()
    if "8" in which:
        fig8()
