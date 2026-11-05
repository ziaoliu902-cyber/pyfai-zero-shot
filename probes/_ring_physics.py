# -*- coding: utf-8 -*-
"""物理终审：真实宽帧环位实测 vs CeO2 理论。
径向剖面 → 峰位 → 用 GT 几何换算 q → 对齐理论 d-spacings。
同时量中心遮挡半径（挡光盘）与 sim 域(≤45px)的差。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI

from real_refine_v6 import match_gain, to_nn_input_wide
from real_probe import load_geom

PXEFF = 172e-6 / (521.0 / 2527)     # 宽画布有效方形像素 ≈ 834µm
LAM = 0.972386e-10
# CeO2 (PDF 34-0394) 主环 d(Å) → q=2π/d
DS = np.array([3.124, 2.705, 1.913, 1.631, 1.562, 1.352, 1.241,
               1.210, 1.104, 1.041, 0.965, 0.914, 0.902, 0.856])


def radial_profile(canvas, poni_px):
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - poni_px[0], xx - poni_px[1])
    nb = 260
    prof = np.zeros(nb)
    cnt = np.zeros(nb)
    idx = (r / (H / nb)).astype(int)
    m = (idx >= 0) & (idx < nb)
    np.add.at(prof, idx[m], canvas[m])
    np.add.at(cnt, idx[m], 1)
    prof /= np.maximum(cnt, 1)
    return prof, (np.arange(nb) + 0.5) * (H / nb) * PXEFF   # r 米


def peaks(prof, r_m, n=8):
    p = prof - np.convolve(prof, np.ones(15) / 15, "same")
    sel = []
    for i in range(3, len(p) - 3):
        if p[i] == max(p[i - 3:i + 4]) and p[i] > 0.05 * p.max() \
                and i > 18:                      # 跳过中心区
            sel.append(i)
    sel = sorted(sel, key=lambda i: -p[i])[:n]
    return sorted([(r_m[i], prof[i]) for i in sel])


for d in (250, 400):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    wide, k, extra = to_nn_input_wide(img)
    wide = match_gain(wide)
    # GT poni 在画布系：行=poni1（有效像素）；列=(poni2-裁列偏移)
    f = 521.0 / img.shape[0]
    poni_r = p.poni1 / PXEFF
    poni_c = (p.poni2 - extra) / PXEFF
    prof, r_m = radial_profile(wide, (poni_r, poni_c))
    pk = peaks(prof, r_m)
    print(f"\n== ceria {d}mm  GT={p.dist*1000:.1f}  "
          f"poni_px=({poni_r:.0f},{poni_c:.0f}) ==")
    # 环位 → 反解 dist：r = D·tan(2θ) → D_i = r_i / tan(2θ_i)
    qs = 2 * np.pi / DS
    th = np.arcsin(qs * LAM / (4 * np.pi))
    t2 = np.tan(2 * th)
    # 匹配：按可见序（q 小→大）对前几个峰
    est = []
    for (r, a), tt in zip(pk[:5], t2[:len(pk[:5])]):
        est.append(r / tt * 1000)
        print(f"  r={r*1000:6.1f}mm 强度{a:9.0f} → "
              f"若为首环则 D={r/tt*1000:7.1f}mm")
    print(f"  前五环首环假设的 D 中位: "
          f"{np.median(sorted(est)[:3]):.1f}mm  (GT {p.dist*1000:.1f})")
    # 中心遮挡：半径内剖面近零的最小 r
    zc = next((r_m[i] for i in range(6, 200)
               if prof[i] < 0.02 * prof.max()), None)
    print(f"  中心低信号半径 ≈ "
          f"{(zc*1000 if zc else 0):.0f}mm ≈ "
          f"{(zc/PXEFF if zc else 0):.0f}px（sim 挡光盘 ≤45px）")
