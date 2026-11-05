# -*- coding: utf-8 -*-
r"""_bm20_sweep.py — rot2 失谐扫描：峰位=0 则模型全域成立；峰位漂移=测 k/非线性。

对 del_15/25/45/55/65 五帧，rot2 偏移 -2.5..+2.5 deg 步 0.25，环带对比度曲线峰位。
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
WL = 7.749012e-11   # placeholder, overwritten below

def read_poni(path):
    import re as _re
    txt = open(path, encoding="utf-8", errors="replace").read()
    return {k: float(v) for k, v in
            _re.findall(r"^(Distance|Poni1|Poni2|Rot1|Rot2|Rot3|Wavelength):\s*(\S+)", txt, _re.M)}

WL = read_poni(os.path.join(BASE, "del_31.5_0001p.poni"))["Wavelength"]

det = Detector(pixel1=PX, pixel2=PX, max_shape=(195, 487))
cal = get_calibrant("LaB6")
ds = np.unique(cal.dspacing)
ds = ds[np.isfinite(ds)]
lam = WL * 1e10
ring_tth = []
for d in ds:
    s = lam / (2 * d)
    if 0 < s < 1:
        ring_tth.append(np.degrees(2 * np.arcsin(s)))
ring_tth = np.array(ring_tth)

OFFS = np.arange(-2.5, 2.51, 0.25)
W, WOUT = 0.35, 1.2

def contrast_at(tag, off_deg):
    f = os.path.join(BASE, "del_%04.1f_0001p.cbf" % tag)
    img = fabio.open(f).data.astype(np.float64)
    img = np.where(img >= 4e9, 0.0, img)
    r2 = R20 + np.radians(tag - 31.5) + np.radians(off_deg)
    ai = AzimuthalIntegrator(D, P1, P2, 0.0, r2, 0.0, detector=det, wavelength=WL)
    tth = ai.array_from_unit(unit="2th_deg")
    cons = []
    for tr in ring_tth:
        on = np.abs(tth - tr) < W
        out = (np.abs(tth - tr) >= W + 0.15) & (np.abs(tth - tr) < W + 0.15 + WOUT) & (tth > 1.0)
        if on.sum() < 50 or out.sum() < 200:
            continue
        cons.append(img[on].mean() / max(img[out].mean(), 1e-6))
    return np.median(cons) if cons else np.nan, len(cons)

print("%-10s %-22s %-10s %s" % ("delta", "contrast curve", "peak_off", "n_rings"))
results = {}
for tag in (15.0, 25.0, 35.0, 45.0, 55.0, 65.0):
    cs = []
    for off in OFFS:
        c, n = contrast_at(tag, float(off))
        cs.append(c)
    cs = np.array(cs, dtype=float)
    ok = np.isfinite(cs)
    if ok.sum() < 5:
        print("%-10s insufficient rings" % tag)
        continue
    ipk = int(np.nanargmax(cs))
    # parabolic refine around peak
    if 0 < ipk < len(cs) - 1:
        y0, y1, y2 = cs[ipk - 1], cs[ipk], cs[ipk + 1]
        denom = (y0 - 2 * y1 + y2)
        dx = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        peak = OFFS[ipk] + dx * 0.25
    else:
        peak = OFFS[ipk]
    curve = " ".join(("%.1f" % c if np.isfinite(c) else " . ") for c in cs[::2])
    print("%-10s %-22s %+7.2f    peakC=%.2f" % (tag, curve, peak, np.nanmax(cs)))
    results[tag] = peak

# drift law
tags = sorted(results)
offs = np.array([results[t] for t in tags])
print("\npeak offsets:", {t: round(results[t], 2) for t in tags})
A = np.stack([np.array(tags), np.ones(len(tags))], 1)
slope, icept = np.linalg.lstsq(A, offs, rcond=None)[0]
print("drift: %.4f deg-off per deg-delta (intercept %.2f)" % (slope, icept))
if abs(slope) < 0.02:
    print("=> k=1 model holds over full range (drift < 0.02 deg/deg)")
else:
    print("=> systematic drift; k correction = %.5f" % (1 + slope))
