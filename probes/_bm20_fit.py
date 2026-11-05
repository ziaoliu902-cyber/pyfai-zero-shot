# -*- coding: utf-8 -*-
r"""_bm20_fit.py — BM20 逐帧 rot2 一维精调（低索引强环, ±1.5° bounded）→ k 线性度判定。

抽样 13 帧（δ=8..64 步 4.5）。每帧 minimize_scalar 目标=前 N_LO 环环带信噪比（带内均值/带外本底, 带外取两侧环形）。
输出 rot2_fit(delta) vs 模型线, 回归 k + 残差。残差 <~0.15 deg → 生成器成立。
"""
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
from pyFAI.calibrant import get_calibrant
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

BASE = r"C:\Users\liu\Desktop\衍射测试图\ext1\LaB6_gonio_BM20"
PX = 172e-6
D, P1, P2, R20 = 0.8056, 0.00563, 0.04612, 0.560707

def read_wl():
    txt = open(os.path.join(BASE, "del_31.5_0001p.poni"), encoding="utf-8", errors="replace").read()
    return float(re.search(r"^Wavelength:\s*(\S+)", txt, re.M).group(1))

WL = read_wl()
det = Detector(pixel1=PX, pixel2=PX, max_shape=(195, 487))
cal = get_calibrant("LaB6")
ds = np.unique(cal.dspacing)
ds = ds[np.isfinite(ds)][::-1]            # descending d = ascending 2th
lam = WL * 1e10
tth_all = np.degrees(2 * np.arcsin(lam / (2 * ds)))
N_LO = 12                                  # low-index strong rings only
tth_use = tth_all[:N_LO]
W = 0.30

def tth_map(rot2):
    ai = AzimuthalIntegrator(D, P1, P2, 0.0, rot2, 0.0, detector=det, wavelength=WL)
    return ai.array_from_unit(unit="2th_deg")

def score(tag, img_cache, rot2):
    img, tth = img_cache, tth_map(rot2)
    tot, cnt = 0.0, 0
    for tr in tth_use:
        on = np.abs(tth - tr) < W
        lo = (tth >= tr - W - 1.0) & (tth < tr - W)
        hi = (tth > tr + W) & (tth <= tr + W + 1.0)
        if on.sum() < 40 or (lo.sum() + hi.sum()) < 150:
            continue
        bkg = (img[lo].sum() + img[hi].sum()) / max(lo.sum() + hi.sum(), 1)
        tot += img[on].mean() / max(bkg, 1e-6)
        cnt += 1
    return (tot / cnt) if cnt else 1.0, cnt

def fit_frame(tag):
    f = os.path.join(BASE, "del_%04.1f_0001p.cbf" % tag)
    img = fabio.open(f).data.astype(np.float64)
    img = np.where(img >= 4e9, 0.0, img)
    r2m = R20 + np.radians(tag - 31.5)
    def neg(off):
        s, _ = score(tag, img, r2m + np.radians(off))
        return -s
    res = minimize_scalar(neg, bounds=(-1.5, 1.5), method="bounded",
                          options=dict(xatol=0.02))
    _, n = score(tag, img, r2m + np.radians(res.x))
    return float(np.degrees(res.x)), n, -res.fun

print("%-8s %-10s %-8s %-8s %s" % ("delta", "fit_off(deg)", "rings", "score", "rot2_fit(deg)"))
tags = [8.0, 12.5, 17.0, 21.5, 26.0, 30.5, 35.0, 39.5, 44.0, 48.5, 53.0, 57.5, 62.0, 65.0]
fit_off, used = [], []
for t in tags:
    off, n, s = fit_frame(t)
    r2f = np.degrees(R20 + np.radians(t - 31.5)) + off
    print("%-8.1f %+10.3f %-8d %-8.2f %8.3f" % (t, off, n, s, r2f))
    fit_off.append(off)
    used.append(n)

fit_off = np.array(fit_off)
ok = np.array(used) >= 4
if ok.sum() >= 3:
    A = np.stack([np.array(tags)[ok] - 31.5, np.ones(ok.sum())], 1)
    slope, icept = np.linalg.lstsq(A, fit_off[ok], rcond=None)[0]
    pred = A @ np.array([slope, icept])
    resid = fit_off[ok] - pred
    print("\nfit_off = %.5f*(delta-31.5) + %.3f ; resid rms = %.3f deg (n=%d)"
          % (slope, icept, float(np.sqrt((resid ** 2).mean())), ok.sum()))
    print("=> k_refined = %.5f (was 1.0); intercept %.3f deg" % (1 + slope, icept))
else:
    print("\ninsufficient frames with >=4 rings in view")
