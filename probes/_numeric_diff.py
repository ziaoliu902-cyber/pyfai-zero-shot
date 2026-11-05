# -*- coding: utf-8 -*-
"""数值解剖：ceria_350 真实 vs GT孪生，逐项量化「皮」。
1) 环位对齐 2) 环宽 3) 方位角调制深度 4) 背景径向形状 5) 残差结构
"""
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
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

LAM = 0.972386
K = 2.425
PXEFF = 172e-6 / (521.0 / 2527)

img = fabio.open(os.path.join(
    ROOT, "data", "real", "MX_ceria/ceria_350_1_0001.cbf")).data
p = pyFAI.load(os.path.join(
    ROOT, "data", "real", "MX_ceria/ceria_350_1_0001.poni"))
D = p.dist
wide, k, extra = to_nn_input_wide(img)
pr = p.poni1 / PXEFF
pc = (p.poni2 - extra) / PXEFF

db = get_scaled_detector("Pilatus_1M", 2, K)
Hm, Wm = db.extent
rng = np.random.default_rng(7)
g = dict(dist=D, poni1=Hm * (pr / 521), poni2=Wm * (pc / 490),
         rot1=0.0, rot2=0.0, rot3=0.0)
cfg = SimConfig()
cfg.wavelength = LAM * 1e-10
cfg.beamstop_prob = 0.0
s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                 pixel_scale=K, geom=g)
twin = match_gain(s["img"].astype(np.float64))


def radial(canvas):
    yy, xx = np.mgrid[0:521, 0:490]
    r = np.hypot(yy - pr, xx - pc)
    nb = 245
    idx = np.clip((r / 2).astype(int), 0, nb - 1)     # 2px/桶
    prof = np.bincount(idx.ravel(), weights=canvas.ravel(), minlength=nb)
    cnt = np.bincount(idx.ravel(), minlength=nb)
    return prof / np.maximum(cnt, 1)


def peaks(prof):
    out = []
    for i in range(3, len(prof) - 3):
        if prof[i] == max(prof[i - 3:i + 4]) and prof[i] > 0.25 * prof.max() \
                and (not out or i - out[-1][0] > 6):
            out.append((i, prof[i]))
    return out


Lr, Lt = np.log1p(wide), np.log1p(twin)
fr, ft = radial(Lr), radial(Lt)
print("== 1) 环位（桶=2px） ==")
prk, ptk = peaks(fr), peaks(ft)
print("真实环位:", [f"{i*2}px" for i, _ in prk[:8]])
print("孪生环位:", [f"{i*2}px" for i, _ in ptk[:8]])
if prk and ptk:
    pairs = []
    for i, a in prk[:8]:
        j = min(ptk, key=lambda t: abs(t[0] - i))
        if abs(j[0] - i) <= 4:
            pairs.append((j[0] - i) * 2)
    print("配对Δpx(孪生-真实):", pairs, " 均值", np.mean(pairs) if pairs else "-")

print("== 2) 环宽（最强峰半高宽，桶2px） ==")


def fwhm(prof, pk):
    i = pk[0]
    h = prof[i] / 2
    l = i
    while l > 0 and prof[l] > h: l -= 1
    r = i
    while r < len(prof) - 1 and prof[r] > h: r += 1
    return (r - l) * 2


print(f"真实 FWHM={fwhm(fr, prk[0])}px  孪生 FWHM={fwhm(ft, ptk[0])}px"
      if prk and ptk else "-")

print("== 3) 方位角调制（r=180±3px 环带，36 槽 CV） ==")
yy, xx = np.mgrid[0:521, 0:490]
rr = np.hypot(yy - pr, xx - pc)
ang = (np.arctan2(yy - pr, xx - pc) + np.pi) / (2 * np.pi) * 36
band = (rr > 177) & (rr < 183)
for nm, L in (("真实", Lr), ("孪生", Lt)):
    v = L[band]
    a = ang[band]
    prof = np.array([v[a == i].mean() for i in range(36)])
    prof = prof[np.isfinite(prof)]
    print(f"  {nm}: CV(σ/μ)={prof.std()/prof.mean():.3f}")

print("== 4) 背景径向形状（log1p 中位，环外桶） ==")
for nm, f in (("真实", fr), ("孪生", ft)):
    med = np.median(f)
    print(f"  {nm}: r50px={f[25]:.2f} r100={f[50]:.2f} r150={f[75]:.2f} "
          f"r200={f[100]:.2f} (中位{med:.2f})")

print("== 5) 残差结构（logR-logT，按半径分带均值/σ） ==")
res = Lr - Lt
for lo, hi in ((0, 40), (40, 80), (80, 120), (120, 160), (160, 200),
               (200, 245)):
    msk = (rr >= lo) & (rr < hi)
    v = res[msk]
    print(f"  r[{lo:>3d},{hi:>3d}): 均值{v.mean():+.3f} σ{v.std():.3f}")
