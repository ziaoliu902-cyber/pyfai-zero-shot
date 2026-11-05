# -*- coding: utf-8 -*-
"""_t5_sharp.py：1D 谱锐度三方对拍——校正有效性最终判决。
P1=ai_spline(raw)（GT 锐基准）；P2=ai_flat(inv)（我们的校正）；
P3=ai_flat(raw)（不校正）。指标：top 峰平均 FWHM(2θ bins) + 峰高。
P2≈P1 → 校正有效；P2≈P3 → 校正无效（图本来就没畸变/别的因）。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(os.path.join(ROOT, "data", "real", "silx_zoo"))

import pyFAI.azimuthalIntegrator
from pyFAI.detectors import Detector
from scipy.signal import find_peaks
from scipy.ndimage import map_coordinates
from zoo_run import load_img

from _t5_inv import undistort   # noqa: E402  (round-trip 已验证)

for img_f, poni_f in [("Frelon2k.edf", "Frelon2k.poni"),
                      ("halfccd.edf", "halfccd.poni")]:
    img = load_img(os.path.join(ROOT, "data", "real", "silx_zoo", img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    inv = undistort(img, ai.detector.spline)
    ai_flat = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(
        dist=ai.dist, poni1=ai.poni1, poni2=ai.poni2, rot1=ai.rot1,
        rot2=ai.rot2, rot3=ai.rot3, wavelength=1e-10,
        detector=Detector(pixel1=ai.detector.pixel1,
                          pixel2=ai.detector.pixel2))
    ai_spl = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(
        dist=ai.dist, poni1=ai.poni1, poni2=ai.poni2, rot1=ai.rot1,
        rot2=ai.rot2, rot3=ai.rot3, wavelength=1e-10,
        detector=ai.detector)
    print(f"\n== {poni_f}")
    for tag, geom, cur in (("P1 spline/raw", ai_spl, img),
                           ("P2 flat/inv ", ai_flat, inv),
                           ("P3 flat/raw", ai_flat, img)):
        tt, I = geom.integrate1d(cur, 4000, unit="2th_deg",
                                 method="splitbbox")
        rng = I.max() - np.median(I)
        pk, props = find_peaks(I, prominence=0.05 * rng,
                               width=1, wlen=41)
        if len(pk) == 0:
            print(f"   {tag}: 0 peaks")
            continue
        w = props["widths"]                      # FWHM in bins
        h = I[pk]
        top = np.argsort(h)[-15:]
        print(f"   {tag}: {len(pk)}峰 FWHM_med={np.median(w[top]):.2f}bins "
              f"高度_med={np.median(h[top]) - np.median(I):.1f} "
              f"锐度=高/宽={np.median((h[top]-np.median(I))/w[top]):.1f}")
