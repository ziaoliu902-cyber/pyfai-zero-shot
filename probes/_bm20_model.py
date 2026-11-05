# -*- coding: utf-8 -*-
r"""_bm20_model.py — BM20 干净模型 121 帧物理检验。

模型（锚定 del_31.5 + del_35.0，绕纯 rot2 轴，k=1）：
  rot2(delta) = rot2(31.5) + radians(delta - 31.5)
  D/poni1/poni2/rot1/rot3/wl 共享常数
检验：对全部 121 帧 cbf，预测 LaB6 环带处亮度 vs 环外本底的对比度。
环带对比度 >> 1 即模型在该 delta 成立。无需重标定——纯预测检验。
"""
import glob
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

BASE = r"C:\Users\liu\Desktop\衍射测试图\ext1\LaB6_gonio_BM20"
PX = 172e-6

def rmat(r1, r2, r3):
    sx, cx = np.sin(r1), np.cos(r1)
    sy, cy = np.sin(r2), np.cos(r2)
    sz, cz = np.sin(r3), np.cos(r3)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def log_so3(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-12:
        return np.zeros(3), 0.0
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * th / (2 * np.sin(th)), th

def read_poni(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    d = dict(re.findall(r"^(Distance|Poni1|Poni2|Rot1|Rot2|Rot3|Wavelength):\s*(\S+)",
                        txt, re.M))
    return {k: float(v) for k, v in d.items()}

T = {t: read_poni(os.path.join(BASE, "del_%s_0001p.poni" % t))
     for t in ("31.5", "33.5", "35.0", "35.5")}
WL = T["31.5"]["Wavelength"]
D = float(np.mean([T[t]["Distance"] for t in ("31.5", "35.0")]))
P1 = float(np.mean([T[t]["Poni1"] for t in ("31.5", "35.0")]))
P2 = float(np.mean([T[t]["Poni2"] for t in ("31.5", "35.0")]))
R20 = T["31.5"]["Rot2"]

R31 = rmat(T["31.5"]["Rot1"], T["31.5"]["Rot2"], T["31.5"]["Rot3"])
R35 = rmat(T["35.0"]["Rot1"], T["35.0"]["Rot2"], T["35.0"]["Rot3"])
w, th = log_so3(R35 @ R31.T)
print("anchor check 31.5->35.0: |w|=%.4f deg (truth 3.5 deg) axis=%s"
      % (np.degrees(th), np.round(w / np.linalg.norm(w), 3)))
print("model: D=%.4f poni1=%.5f poni2=%.5f rot2_0=%.6f wl=%.4e"
      % (D, P1, P2, R20, WL))

det = Detector(pixel1=PX, pixel2=PX, max_shape=(195, 487))
cal = get_calibrant("LaB6")
ds = np.unique(cal.dspacing)
ds = ds[np.isfinite(ds)]

def frame_model_ai(delta_deg):
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    r2 = R20 + np.radians(delta_deg - 31.5)
    return AzimuthalIntegrator(D, P1, P2, 0.0, r2, 0.0,
                               detector=det, wavelength=WL)

files = sorted(glob.glob(os.path.join(BASE, "del_*_0001p.cbf")))
print("\nframes found:", len(files))
W = 0.35
WOUT = 1.2
rows = []
for f in files:
    tag = float(re.search(r"del_([\d.]+)_0001p", f).group(1))
    img = fabio.open(f).data.astype(np.float64)
    img = np.where(img >= 4e9, 0.0, img)
    ai = frame_model_ai(tag)
    tth = ai.array_from_unit(unit="2th_deg")
    lam = WL * 1e10
    contrasts = []
    for d in ds:
        s = lam / (2.0 * d)
        if not (0 < s < 1):
            continue
        tr = np.degrees(2 * np.arcsin(s))
        on = np.abs(tth - tr) < W
        out = (np.abs(tth - tr) >= W + 0.15) & (np.abs(tth - tr) < W + 0.15 + WOUT) & (tth > 1.0)
        if on.sum() < 50 or out.sum() < 200:
            continue
        contrasts.append(img[on].mean() / max(img[out].mean(), 1e-6))
    rows.append((tag, np.median(contrasts) if contrasts else np.nan, len(contrasts)))

tags = np.array([r[0] for r in rows])
cons = np.array([r[1] for r in rows])
nrng = np.array([r[2] for r in rows])
good = np.isfinite(cons) & (cons > 1.5)
print("contrast>1.5 frames: %d/%d ; median contrast = %.2f"
      % (good.sum(), len(rows), np.nanmedian(cons)))
print("\n%-14s %-10s %-8s" % ("delta bin", "median C", "frames"))
for lo in np.arange(5, 66, 6.0):
    m = (tags >= lo) & (tags < lo + 6)
    if m.any():
        print("%4.0f-%4.0f     %8.2f  %3d" % (lo, lo + 6, np.nanmedian(cons[m]), m.sum()))

idx = np.argsort(np.nan_to_num(cons, nan=-1))[:8]
print("\nworst frames:")
for i in idx:
    print("  del_%04.1f C=%.2f nrings=%d" % (tags[i], cons[i], nrng[i]))

md = ["# BM20 干净模型 121 帧物理检验（rot2 线性 + 共享 dist/poni）", "",
      "锚定 del_31.5+del_35.0（k=1.003 绕纯 rot2 轴；33.5 半步异常/35.5 rot1 异常帧剔除锚定）",
      "模型: rot2(delta)=rot2_31.5+rad(delta-31.5), D=%.4f, poni=(%.5f,%.5f), wl=%.4e" % (D, P1, P2, WL), "",
      "| delta | contrast | rings |", "|---|---|---|"]
for t, c, n in rows:
    md.append("| %.1f | %.2f | %d |" % (t, c, n))
md += ["", "对比度>1.5: %d/%d 帧, 中位 %.2f" % (good.sum(), len(rows), np.nanmedian(cons))]
with open(os.path.join(ROOT, "docs", "bm20_model_check.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")
print("\nmd -> docs/bm20_model_check.md")
