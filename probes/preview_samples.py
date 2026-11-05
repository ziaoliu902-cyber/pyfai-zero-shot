# -*- coding: utf-8 -*-
"""样本可视化验货：3x3 contact sheet + 单张全分辨率 PNG，叠加解析环半径虚线圈。

环心 = poni（未倾斜近似圆；小倾角下近似成立，倾斜量见标题 rot）。
输出：preview/contact_sheet.png 与 preview/sample_XX_<phase>.png
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

from torchfai.config import SimConfig
from torchfai.simulator import (get_detector, synth_sample, ring_radius_m,
                                phase_qi, make_geometry)
from torchfai.phases import PHASE_NAMES

OUT = Path(__file__).resolve().parent.parent / "preview"
OUT.mkdir(exist_ok=True)

cfg = SimConfig()
db = get_detector(cfg.detector_name, cfg.binning)
lam_A = cfg.wavelength * 1e10
rng = np.random.default_rng(7)

# ---- 采 9 张：优先集齐 6 个相（每相首张），之后任意相补位 ----
picks, seen = [], set()
for _ in range(80):
    s = synth_sample(rng, cfg, db)
    ph = s["phase"]
    if len(picks) < 9 and (ph not in seen or len(seen) == len(PHASE_NAMES)):
        picks.append(s)
        seen.add(ph)
    if len(picks) >= 9:
        break

def panel(ax, s, db, title_only=False):
    img = s["img"].astype(np.float32)
    g, ph = s["geom"], s["phase"]
    bs_r = float(s["aux"][4])
    px = db.pixel
    cy, cx = g["poni1"] / px, g["poni2"] / px   # poni 像素坐标（行, 列）

    ax.imshow(img, origin="lower", cmap="magma",
              norm=matplotlib.colors.PowerNorm(0.4), interpolation="nearest")
    # ---- 精确环位置：q-map 等值线（含全部旋转/倾斜效应，与生成图像同一 Geometry）----
    gobj = make_geometry(g, cfg, db)
    qm = gobj.array_from_unit(shape=db.shape, typ="center", unit="q_A^-1", scale=True)
    qm = np.nan_to_num(qm, nan=100.0, posinf=100.0)
    qs, Is = phase_qi(ph)
    # 可见环判定（与 simulator 同一近似，仅用于选层和计数）
    rr_px = ring_radius_m(qs, g["dist"], lam_A) / px
    Hm, Wm = db.extent
    corners = np.array([g["poni1"], Hm - g["poni1"], g["poni2"], Wm - g["poni2"]])
    r_far = float(np.hypot(max(corners[0], corners[1]), max(corners[2], corners[3])))
    thr = (bs_r + 3.0) if bs_r > 0 else 0.0
    vis = (rr_px > thr) & (rr_px < 1.05 * r_far / px)
    levels = sorted({float(q0) for q0, v in zip(qs, vis) if v})
    if levels:
        ax.contour(qm, levels=levels, colors="white", linewidths=0.7,
                   linestyles="--", alpha=0.75)
    if bs_r > 0:
        ax.add_patch(Circle((cx, cy), bs_r, fill=False, ls="-", lw=0.7,
                            ec="red", alpha=0.8))
    rt = np.degrees([g["rot1"], g["rot2"], g["rot3"]])
    ax.set_title(f"{ph}  d={g['dist']:.3f}m  poni=({g['poni1']*1e3:.0f},{g['poni2']*1e3:.0f})mm\n"
                 f"rot=({rt[0]:+.1f},{rt[1]:+.1f},{rt[2]:+.1f})deg  vis={int(vis.sum())}  "
                 f"bs={bs_r:.0f}px  max={int(img.max())}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

# ---- 单张 ----
for i, s in enumerate(picks):
    fig, ax = plt.subplots(figsize=(7, 6.8), dpi=140)
    panel(ax, s, db)
    fig.tight_layout()
    f = OUT / f"sample_{i:02d}_{s['phase']}.png"
    fig.savefig(f); plt.close(fig)
    print("saved", f.name)

# ---- contact sheet ----
fig, axes = plt.subplots(3, 3, figsize=(16.5, 14.5), dpi=110)
for ax, s in zip(axes.ravel(), picks):
    panel(ax, s, db)
fig.suptitle(f"torchfai Week1 simulator samples  |  {cfg.detector_name} binned{cfg.binning}"
             f" {db.shape[1]}x{db.shape[0]} px={db.pixel*1e6:.0f}um  Cu Ka1  "
             f"white dashed=exact q-contours, red=beamstop", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
f = OUT / "contact_sheet_v2.png"
fig.savefig(f); plt.close(fig)
print("saved", f.name)
