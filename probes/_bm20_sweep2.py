# -*- coding: utf-8 -*-
r"""_bm20_sweep2.py — BM20 中段窄域失谐扫描（全环动态可见性 + 中位数聚合）。

对 del_12/17/22/27/32 五帧：rot2 偏移 ±0.8 deg 步 0.1，每点取「画布内可见环」
的环带对比度中位数。峰位 ≈0 → k=1 模型在该 δ 成立。
聚合用中位数 + 每环带内像素门（>=40）+ 双侧本底，抗邻环串扰。
"""
import os
import re
import sys

import numpy as np

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
cal = get_calant = get_calibrant("LaB6")
ds = np.unique(cal.dspacing)
ds = ds[(np.isfinite(ds)) & (ds > 0)]
lam = WL * 1e10
s = lam / (2 * ds)
m = s < 1
tth_rings = np.degrees(2 * np.arcsin(s[m]))
W = 0.25

def tth_map(rot2):
    ai = AzimuthalIntegrator(D, P1, P2, 0.0, rot2, 0.0, detector=det, wavelength=WL)
    return ai.array_from_unit(unit="2th_deg")

def med_contrast(img, rot2):
    tth = tth_map(rot2)
    cons = []
    for tr in tth_rings:
        on = np.abs(tth - tr) < W
        lo = (tth >= tr - W - 0.9) & (tth < tr - W)
        hi = (tth > tr + W) & (tth <= tr + W + 0.9)
        if on.sum() < 40 or (lo.sum() + hi.sum()) < 120:
            continue
        bkg = (img[lo].sum() + img[hi].sum()) / max(lo.sum() + hi.sum(), 1)
        cons.append(img[on].mean() / max(bkg, 1e-6))
    return (float(np.median(cons)) if cons else np.nan), len(cons)

OFFS = np.round(np.arange(-0.8, 0.81, 0.1), 2)
print("%-7s | %s" % ("delta", " ".join("%+4.1f" % o for o in OFFS[::2])))
peaks = {}
for tag in (12.0, 17.0, 22.0, 27.0, 32.0):
    f = os.path.join(BASE, "del_%04.1f_0001p.cbf" % tag)
    img = fabio.open(f).data.astype(np.float64)
    img = np.where(img >= 4e9, 0.0, img)
    r2m = R20 + np.radians(tag - 31.5)
    cs, ns = [], []
    for off in OFFS:
        c, n = med_contrast(img, r2m + np.radians(float(off)))
        cs.append(c)
        ns.append(n)
    cs = np.array(cs, dtype=float)
    if not np.isfinite(cs).any():
        print("%-7.1f | no rings" % tag)
        continue
    curve = " ".join(("%4.1f" % c if np.isfinite(c) else "  . ") for c in cs[::2])
    ipk = int(np.nanargmax(cs))
    print("%-7.1f | %s   peak=%+.2f (C=%.2f, nrings=%d)" % (tag, curve, OFFS[ipk], cs[ipk], ns[ipk]))
    peaks[tag] = float(OFFS[ipk])

if peaks:
    print("\npeak offsets:", {t: round(p, 2) for t, p in peaks.items()})
    v = np.array(list(peaks.values()))
    print("median |peak| = %.2f deg, max = %.2f deg" % (np.median(np.abs(v)), np.abs(v).max()))
