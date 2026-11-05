# -*- coding: utf-8 -*-
r"""_nc_duel2.py — zoo 关键帧四中心锐度对决（审计口径复核）。

ceria 翻案教训：带倾角 GT 的 Fit2D 中心可能比共心 ref 更锐。 zoo 判决
（halfccd 56px / E9M 5.8px）是否也站得住？对每个关键帧：
  GT-naive / GT-Fit2D / ours(SolveWorker) / 共心ref 四中心，
  前 3 条可评环的径向谱峰高——最锐者胜。
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

import pyFAI

from gui_solver import load_any, SolveWorker
from solver_core import concentric_center
from torchfai.phases import phase_qs

ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
CASES = [
    ("P2M", "Pilatus2M.cbf", "Pilatus2M.poni"),
    ("E9M", "Eiger9M.h5", "Eiger9M.poni"),
    ("halfccd", "halfccd.edf", "halfccd.poni"),
    ("mar3450", "LaB6_260210.mar3450", "Mar3450.poni"),
]


def profile(img, cx, cy, dr=0.25):
    rmax = min(max(img.shape) * 1.2, 4000)
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
    if m.sum() < 5:
        return 0.0
    seg = pp[m]
    return float(seg.max() - np.percentile(seg, 10))


for tag, img_f, poni_f in CASES:
    os.chdir(ZOO)                       # spline 相对路径修复
    p = pyFAI.load(os.path.join(ZOO, poni_f))
    px_um = float(p.detector.pixel1) * 1e6
    px = float(p.detector.pixel1)
    lam = p.wavelength * 1e10
    naive = (p.poni2 / float(p.detector.pixel2), p.poni1 / px)
    f2d = p.getFit2D()
    gt2d = (f2d["centerX"], f2d["centerY"])
    img = load_any(os.path.join(ZOO, img_f))
    img = np.where(img >= 4e9, 0.0, img)
    box = {}
    w = SolveWorker(img, lam, px_um)
    w.done.connect(lambda r: box.update(r=r))
    w.run()
    if not box.get("r"):
        print(f"== {tag} == SOLVE FAIL")
        continue
    res = box["r"]
    ours = (res["cx"], res["cy"])
    ph = res["ph"]
    cr = concentric_center(img, ours, res["D_px"], phase_qs(ph), lam)
    qs = phase_qs(ph)[:10]
    tt = 2 * np.arcsin(np.clip(lam * qs / (4 * np.pi), 0, 0.9999))
    radii = res["D_px"] * np.tan(tt)
    r_edge = 0.9 * min(ours[0], ours[1], img.shape[1] - ours[0],
                       img.shape[0] - ours[1])
    radii = [r for r in radii if 60 < r < r_edge][:3]
    if cr is None:
        ref, refn = (float("nan"), float("nan")), 0
    else:
        ref, refn = (cr["cx"], cr["cy"]), cr["n"]
    print(f"== {tag} ==  相={ph}  rot=({np.degrees(p.rot1)*1000:.0f},"
          f"{np.degrees(p.rot2)*1000:.0f})mdeg  ref环{refn}")
    for ctag, c in (("GT-naive", naive), ("GT-Fit2D", gt2d),
                    ("ours", ours), ("共心ref", ref)):
        if np.isnan(c[0]):
            continue
        pr, pp = profile(img, c[0], c[1])
        pk = [ring_sharp(pr, pp, r0) for r0 in radii]
        print(f"  {ctag:10} ({c[0]:8.1f},{c[1]:8.1f})  峰高="
              f"{[f'{x:.0f}' for x in pk]}")
    print()
