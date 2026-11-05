# -*- coding: utf-8 -*-
"""_t5_diag2.py：Frelon2k 径向谱贫环之谜——图像统计 + pyFAI 1D 谱峰数
vs ring_radii 均值谱峰数对拍（同 GT 几何）。"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import pyFAI.azimuthalIntegrator
from zoo_run import load_img
from solver_core import ring_radii, to_canvas

img = load_img(os.path.join(ZOO, "Frelon2k.edf"))
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload("Frelon2k.poni")
p = np.percentile(img, [1, 50, 90, 99, 99.9, 100])
print("img pct 1/50/90/99/99.9/100:", np.round(p, 1))
print("frac>10*med:", float((img > 10 * np.median(img)).mean()))

# pyFAI 1D（GT 几何含 spline；无 λ → 哑 1Å + 2θ 单位绕开）
ai.wavelength = 1e-10
q, I = ai.integrate1d(img, 2000, unit="2th_deg", method="splitbbox")
from scipy.signal import find_peaks
pk, props = find_peaks(I, prominence=0.02 * (I.max() - np.median(I)))
print(f"pyFAI 1D: {len(pk)} peaks (prom 2%)  "
      f"I range {I.min():.1f}~{I.max():.1f} med {np.median(I):.1f}")

# 我们：GT 中心画布均值谱
canvas, s, _, _ = to_canvas(img)
f2d = ai.getFit2D()
gcx, gcy = f2d["centerX"] / s, f2d["centerY"] / s
peaks = ring_radii(canvas, gcx, gcy)[0]
print(f"ring_radii: {len(peaks)} rings at GT center on canvas s={s}")
# 也试原生
peaks_n = ring_radii(img, float(f2d["centerX"]), float(f2d["centerY"]))[0]
print(f"ring_radii: {len(peaks_n)} rings at GT center on native")

# 对比 halfccd
img2 = load_img(os.path.join(ZOO, "halfccd.edf"))
ai2 = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload("halfccd.poni")
ai2.wavelength = 1e-10
q2, I2 = ai2.integrate1d(img2, 2000, unit="2th_deg", method="splitbbox")
pk2, _ = find_peaks(I2, prominence=0.02 * (I2.max() - np.median(I2)))
canvas2, s2, _, _ = to_canvas(img2)
f2d2 = ai2.getFit2D()
pk_ours = ring_radii(canvas2, f2d2["centerX"] / s2, f2d2["centerY"] / s2)[0]
print(f"halfccd: pyFAI {len(pk2)} peaks / ring_radii {len(pk_ours)} rings")
