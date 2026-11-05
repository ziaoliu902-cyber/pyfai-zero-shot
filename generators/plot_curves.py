# -*- coding: utf-8 -*-
"""从 history.csv 画论文级学习曲线 → runs/<tag>/curves.png

用法：
    python plot_curves.py --tag base_v1            # 默认读 runs/base_v1/history.csv
    python plot_curves.py --tag base_v1 --out runs/base_v1/curves.pdf

左图：train/val 损失（symlog，可显示负的 NLL）+ 学习率（右轴，log）
右图：六参数物理单位 MAE（log 轴）
英文标注，300 dpi，可直接进论文初稿。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAM_NAMES = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
PARAM_UNITS = ("m", "m", "m", "rad", "rad", "rad")


def load_history(tag: str):
    fp = os.path.join("runs", tag, "history.csv")
    rows = []
    with open(fp, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    assert rows, f"{fp} 没有数据行"
    ep = np.array([int(r["epoch"]) for r in rows])
    tr = np.array([float(r["train_loss"]) for r in rows])
    va = np.array([float(r["val_loss"]) for r in rows])
    lr = np.array([float(r["lr"]) for r in rows])
    mae = np.array([[float(r[f"mae_{n}"]) for n in PARAM_NAMES] for r in rows])
    return ep, tr, va, lr, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base_v1")
    ap.add_argument("--out", default=None, help="默认 runs/<tag>/curves.png")
    a = ap.parse_args()

    ep, tr, va, lr, mae = load_history(a.tag)
    out = a.out or os.path.join("runs", a.tag, "curves.png")

    # ---- 左：损失（symlog 能画负值）+ lr 右轴 ----
    # ---- 每变量一面板：损失大图 + lr + 六参数 MAE 独立小图 ----
    fig = plt.figure(figsize=(14, 8), dpi=300)
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.45)

    ax = fig.add_subplot(gs[0:2, 0:2])
    ax.plot(ep, tr, "-", lw=1.6, color="#4C72B0", label="train")
    ax.plot(ep, va, "s", ms=3, lw=1.2, color="#C44E52", label="val")
    best_i = int(np.nanargmin(va))
    ax.plot(ep[best_i], va[best_i], "*", ms=14, color="gold",
            mec="k", mew=0.6, zorder=5, label=f"best (ep {ep[best_i]})")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (NLL, symlog)")
    ax.set_title("Training / validation loss")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    panels = [(gs[0, 2], "lr", lr, "Learning rate", True),
              (gs[0, 3], PARAM_NAMES[0], mae[:, 0], "MAE dist (m)", True),
              (gs[1, 2], PARAM_NAMES[1], mae[:, 1], "MAE poni1 (m)", True),
              (gs[1, 3], PARAM_NAMES[2], mae[:, 2], "MAE poni2 (m)", True),
              (gs[2, 0], PARAM_NAMES[3], mae[:, 3], "MAE rot1 (rad)", True),
              (gs[2, 1], PARAM_NAMES[4], mae[:, 4], "MAE rot2 (rad)", True),
              (gs[2, 2], PARAM_NAMES[5], mae[:, 5], "MAE rot3 (rad)", True)]
    for slot, nm, y, ttl, logy in panels:
        axp = fig.add_subplot(slot)
        axp.plot(ep, y, "o-", ms=2.5, lw=1.2,
                 color="#55A868" if nm == "lr" else "#4C72B0")
        if logy and np.min(y) > 0:
            axp.set_yscale("log")
        axp.set_title(ttl, fontsize=9)
        axp.set_xlabel("Epoch", fontsize=8)
        axp.tick_params(labelsize=7)
        axp.grid(alpha=0.3, which="both")
        axp.annotate(f"{y[-1]:.3g}", xy=(ep[-1], y[-1]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")

    fig.suptitle(f"Learning curves — {a.tag} (per-variable panels)", y=0.995)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved: {out}")
    print(f"best val epoch {ep[best_i]}  val={va[best_i]:.3f}")
    print("final MAE:", {n: f"{mae[-1, j]:.5g}" for j, n in enumerate(PARAM_NAMES)})


if __name__ == "__main__":
    main()
