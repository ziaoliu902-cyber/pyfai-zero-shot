# -*- coding: utf-8 -*-
r"""_nc_ref_duel.py — 中心口径仲裁（ceria）：四种中心谁让环最锐。

  GT-naive : poni1/2 ÷ px（无倾角近似——昨判决实验用的）
  GT-Fit2D : ai.getFit2D()（带倾角换算——审计用的）
  ref      : 共心真值（本次审计）
  ours     : SolveWorker 解
判据：0.25px 径向谱，前 3 条环的峰高（越高=越锐=中心越对）。
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from PyQt5.QtWidgets import QApplication
app = QApplication([])
import i18n
i18n.set_lang("zh")

import fabio
import pyFAI

from gui_solver import load_any, SolveWorker
from solver_core import concentric_center
from torchfai.phases import phase_qs

PX = 172e-6


def profile(img, cx, cy, dr=0.25, rmax=900):
    H, W = img.shape
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - cy, xx - cx)
    bins = np.arange(0, rmax + dr, dr)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    i, v = idx[ok], img.ravel()[ok]
    cnt = np.bincount(i, minlength=len(bins) - 1)
    p = np.bincount(i, weights=v, minlength=len(bins) - 1) / np.maximum(cnt, 1)
    return bins[:-1] + dr / 2, p


def ring_sharp(pr, pp, r0, half=6.0):
    m = np.abs(pr - r0) < half
    seg = pp[m]
    base = np.percentile(seg, 10)
    return float(seg.max() - base)


for d in (150, 250):
    f = os.path.join(ROOT, "data", "real", "MX_ceria",
                     f"ceria_{d}_1_0001.cbf")
    img = load_any(f)
    p = pyFAI.load(f.replace(".cbf", ".poni"))
    naive = (p.poni2 / PX, p.poni1 / PX)
    f2d = p.getFit2D()
    gt2d = (f2d["centerX"], f2d["centerY"])
    box = {}
    w = SolveWorker(img, 0.972386, 172.0)
    w.done.connect(lambda r: box.update(r=r))
    w.run()
    res = box["r"]
    ours = (res["cx"], res["cy"])
    cr = concentric_center(img, ours, res["D_px"], phase_qs("CeO2")[:8],
                           0.972386)
    ref = (cr["cx"], cr["cy"])
    lam, D_px = 0.972386, res["D_px"]
    qs = phase_qs("CeO2")[:8]
    tt = 2 * np.arcsin(np.clip(lam * qs / (4 * np.pi), 0, 0.999))
    radii = D_px * np.tan(tt)
    print(f"== ceria_{d} ==  rot=({np.degrees(p.rot1)*1000:+.1f},"
          f"{np.degrees(p.rot2)*1000:+.1f})mrad")
    for tag, c in (("GT-naive(poni/px)", naive), ("GT-Fit2D", gt2d),
                   ("共心ref", ref), ("ours", ours)):
        pr, pp = profile(img, c[0], c[1])
        pk = [ring_sharp(pr, pp, r0) for r0 in radii[:3] if 60 < r0 < 850]
        print(f"  {tag:18} ({c[0]:8.2f},{c[1]:8.2f})  前3环峰高="
              f"{[f'{x:.0f}' for x in pk]}")
    print()
