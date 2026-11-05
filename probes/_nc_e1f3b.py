# -*- coding: utf-8 -*-
r"""_nc_e1f3b.py — E1f3 B 前哨：非 native 帧（Eiger4M / ceria_400）上
V3-canvas（D=solver + 画布 tf 的 poni/rot 成对）vs 现行 V1（平面中心+借rot）。

判决点：Eiger4M 现行 42px/锐度4% 是否被 V3-canvas 修复；ceria_400 现行
0.095°/+0.11mm 是否不劣化（若改善则连基线一起升级）。
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
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.detectors import Detector

from gui_solver import load_any, SolveWorker, hybrid_ai
from torchfai.phases import phase_qs

ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")


def sigma_sharp(ai, img, qs):
    q, I = ai.integrate1d(img, 3000, unit="q_A^-1", method="nosplit_csr",
                          dummy=-1)
    tot = 0.0
    for qc in qs:
        m = np.abs(q - qc) < 0.02
        if m.sum() < 5:
            continue
        tot += float(I[m].max() - np.percentile(I[m], 10))
    return tot


def dev_max(ai, img, lam, qs):
    tth, I = ai.integrate1d(img, 16000, unit="2th_deg",
                            method="bbox,splitpixel")
    med = float(np.median(I))
    thr = med + 0.05 * (I.max() - med)
    devs = []
    for q in qs:
        t_pred = np.degrees(2 * np.arcsin(q * lam / (4 * np.pi)))
        if not (tth[0] + 0.5 < t_pred < tth[-1] - 0.5):
            continue
        win = (tth > t_pred - 0.20) & (tth < t_pred + 0.20)
        wt, wi = tth[win], I[win]
        if wt.size < 5 or wi.max() <= thr:
            continue
        k = int(np.argmax(wi))
        if k < 4 or k > len(wi) - 5:
            continue
        t_meas = wt[k]
        if 0 < k < len(wi) - 1:
            y0, y1, y2 = wi[k - 1], wi[k], wi[k + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                t_meas = wt[k] + 0.5 * (y0 - y2) / den * (wt[1] - wt[0])
        devs.append(abs(t_meas - t_pred))
    return max(devs) if devs else float("nan")


CASES = [
    ("Eiger4M", os.path.join(ZOO, "Eiger4M.edf"),
     os.path.join(ZOO, "Eiger4M.poni")),
    ("ceria_400", os.path.join(ROOT, "data", "real", "MX_ceria",
                               "ceria_400_1_0001.cbf"),
     os.path.join(ROOT, "data", "real", "MX_ceria",
                  "ceria_400_1_0001.poni")),
]

for tag, img_f, poni_f in CASES:
    os.chdir(os.path.dirname(poni_f))
    gt = pyFAI.load(poni_f)
    g2 = gt.getFit2D()
    px_um = float(gt.detector.pixel1) * 1e6
    lam = (gt.wavelength * 1e10) if gt.wavelength else 0.972386
    img = load_any(img_f)
    img = np.where(img >= 4e9, 0.0, img)

    box = {}
    w = SolveWorker(img, lam, px_um)
    w.done.connect(lambda r: box.update(r=r))
    w.run()
    res = box.get("r")
    tf = res.get("tf") if res.get("tilt_on") else None
    px_m = px_um * 1e-6
    qs = phase_qs(res["ph"])
    qs_s = [q for q in qs if 1.0 < q < 3.6][:6]

    print(f"== {tag} == ph={res['ph']} native={res.get('native')} "
          f"tilt_on={res.get('tilt_on')} "
          + (f"imp={tf['imp']*100:.0f}%" if tf else "tf=None"), flush=True)

    variants = {"V1现行": hybrid_ai(res, px_um)}
    if tf:
        variants["V3-canvas"] = AzimuthalIntegrator(
            dist=res["D_px"] * px_m, poni1=tf["poni1"], poni2=tf["poni2"],
            pixel1=px_m, pixel2=px_m, rot1=tf["rot1"], rot2=tf["rot2"],
            rot3=0., wavelength=res["lam"] * 1e-10)
    variants["GT"] = gt
    for vtag, ai in variants.items():
        f2 = ai.getFit2D()
        dc = (float(np.hypot(f2["centerX"] - g2["centerX"],
                             f2["centerY"] - g2["centerY"]))
              if vtag != "GT" else 0.0)
        dD = ai.dist * 1000 - gt.dist * 1000
        sg = sigma_sharp(ai, img, qs_s)
        dv = dev_max(ai, img, lam, qs)
        print(f"  {vtag:10} dev={dv:6.3f}° Σ={sg:9.3e} ΔD={dD:+7.2f}mm "
              f"中心vsGT={dc:6.2f}px", flush=True)
    print(flush=True)
