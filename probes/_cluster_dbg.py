# -*- coding: utf-8 -*-
"""_cluster_dbg.py：400/450 档 D 投票簇 + NN中心检测峰 dump。"""
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

from ring_fit import (LAM, PXEFF, Q_HKL, HKLS, nn_coarse, r_model,
                      ring_radii)

for d in (400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    print(f"\n===== {d}mm GT_D_px={p.dist/PXEFF:.1f} =====")
    p6, wide, k = nn_coarse(img)
    Hm, Wm = wide.shape
    cx0 = p6[2] / PXEFF * (Wm / 490.0)
    cy0 = p6[1] / PXEFF * (Hm / 521.0)
    D_px0 = p6[0] / PXEFF * (Hm / 521.0)
    print(f"NN中心=({cx0:.1f},{cy0:.1f}) D0={D_px0:.1f}px")
    peaks = ring_radii(wide, cx0, cy0)
    print(f"峰({len(peaks)}):", " ".join(f"{rp:.1f}/{h:.0f}" for rp, h in peaks))
    votes = []
    for rp, hgt in peaks:
        for (h_, k_, l_), q in zip(HKLS, Q_HKL):
            tt = 2 * np.arcsin(np.clip(q * LAM / (4 * np.pi), 0, 0.999))
            D = rp / np.tan(tt)
            if 0.35 * D_px0 < D < 2.8 * D_px0:
                votes.append((D, rp, f"({h_}{k_}{l_})", hgt))
    votes.sort()
    used = [False] * len(votes)
    print("簇:")
    for i, (D, rp, hk, hgt) in enumerate(votes):
        if used[i]:
            continue
        j = i
        mem = []
        while j < len(votes) and votes[j][0] < D * 1.03:
            mem.append(votes[j]); used[j] = True
            j += 1
        if len(mem) >= 3:
            print(f"  D~{np.mean([m[0] for m in mem]):.1f} ({len(mem)}票): "
                  + " ".join(f"{m[2]}@{m[1]:.0f}px(h{m[3]:.0f})" for m in mem))
    # GT 模型半径对照
    cy_gt = p.poni1 / PXEFF * (Hm / 521.0)
    cx_gt = p.poni2 / PXEFF * (Wm / 490.0)
    D_gt = p.dist / PXEFF * (Hm / 521.0)
    print("GT模型半径:", " ".join(
        f"({h_}{k_}{l_})={r_model(np.array([q]), cx_gt, cy_gt, D_gt)[0]:.1f}"
        for (h_, k_, l_), q in zip(HKLS, Q_HKL)))
