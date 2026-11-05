# -*- coding: utf-8 -*-
r"""_nc_duel3.py — halfccd / mar3450 四中心锐度对决（补跑，带栈）。"""
import os
import sys
import traceback

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


def profile(img, cx, cy, dr=0.25):
    rmax = min(max(img.shape) * 1.2, 4000)
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - cy, xx - cx)
    bins = np.arange(0, rmax + dr, dr)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    i, v = idx[ok], img.ravel()[ok]
    cnt = np.bincount(i, minlength=len(bins) - 1)
    p = np.bincount(i, weights=v, minlength=len(bins) - 1) / np.maximum(cnt, 1)
    return bins[:-1] + dr / 2, p


def sharp(pr, pp, r0, half=6.0):
    m = np.abs(pr - r0) < half
    if m.sum() < 5:
        return 0.0
    return float(pp[m].max() - np.percentile(pp[m], 10))


for tag, img_f, poni_f in [
        ("halfccd", "halfccd.edf", "halfccd.poni"),
        ("mar3450", "LaB6_260210.mar3450", "Mar3450.poni")]:
    try:
        os.chdir(ZOO)
        p = pyFAI.load(os.path.join(ZOO, poni_f))
        px_um = float(p.detector.pixel1) * 1e6
        lam = p.wavelength * 1e10
        naive = (p.poni2 / float(p.detector.pixel2),
                 p.poni1 / float(p.detector.pixel1))
        f2d = p.getFit2D()
        gt2d = (f2d["centerX"], f2d["centerY"])
        img = load_any(os.path.join(ZOO, img_f))
        img = np.where(img >= 4e9, 0.0, img)
        box = {}
        w = SolveWorker(img, lam, px_um)
        w.done.connect(lambda r: box.update(r=r))
        w.run()
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
        ref_s = "None" if cr is None else (
            f"({cr['cx']:.1f},{cr['cy']:.1f},n={cr['n']})")
        print(f"== {tag} ==  相={ph}  "
              f"rot=({np.degrees(p.rot1)*1000:.0f},"
              f"{np.degrees(p.rot2)*1000:.0f})mdeg  ref={ref_s}",
              flush=True)
        for ctag, c in (("GT-naive", naive), ("GT-Fit2D", gt2d),
                        ("ours", ours),
                        ("共心ref", (cr["cx"], cr["cy"]) if cr
                         else (float("nan"),) * 2)):
            if np.isnan(c[0]):
                continue
            pr, pp = profile(img, c[0], c[1])
            pk = [f"{sharp(pr, pp, r0):.0f}" for r0 in radii]
            print(f"  {ctag:10} ({c[0]:8.1f},{c[1]:8.1f})  峰高={pk}",
                  flush=True)
    except Exception:
        print(f"== {tag} == EXC:", flush=True)
        traceback.print_exc()
