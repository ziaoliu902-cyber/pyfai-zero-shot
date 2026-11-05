# -*- coding: utf-8 -*-
r"""_nc_tilt_sharp.py — E1f：fit_tilt 进中心路径的锐度复检。

问题：倾角帧上平面解中心距最锐处 5–7px（GT-Fit2D 最锐）。
检验：给平面解追加 fit_tilt（控制点 5 参 LSQ）后，全几何 2θ 谱线锐度
是否追平/超过 GT 几何？三几何同图同口径 integrate1d：
  ours-planar  /  ours-tilt(fit_tilt)  /  GT(poni)
锐度 = CeO2 各线位 ±0.02Å⁻¹ 窗内 (max − p10) 之和 Σ。
另报 ours-tilt 的等效 Fit2D 中心 vs GT-Fit2D 中心距离。
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
from pyFAI.detectors import Detector

from gui_solver import load_any, SolveWorker
from poni_out import fit_tilt
from torchfai.phases import phase_qs

PX = 172e-6
LAM = 0.972386
CERIA = os.path.join(ROOT, "data", "real", "MX_ceria")


def mk_ai(dist, poni1, poni2, rot1, rot2):
    return pyFAI.AzimuthalIntegrator(
        dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=0.0,
        wavelength=LAM * 1e-10,
        detector=Detector(pixel1=PX, pixel2=PX))


def prof_sharpness(ai, img, qs):
    q, I = ai.integrate1d(img, 3000, unit="q_A^-1",
                          method="nosplit_csr", dummy=-1)
    tot = 0.0
    pks = []
    for qc in qs:
        m = np.abs(q - qc) < 0.02
        if m.sum() < 5:
            continue
        s = float(I[m].max() - np.percentile(I[m], 10))
        pks.append(s)
        tot += s
    return tot, pks


qs_all = np.asarray(phase_qs("CeO2"), float)
qs_use = [q for q in qs_all if 1.4 < q < 3.4][:6]

for d in (150, 250, 300):
    f = os.path.join(CERIA, f"ceria_{d}_1_0001.cbf")
    img = load_any(f)
    img = np.where(img >= 4e9, 0.0, img)
    gt = pyFAI.load(f.replace(".cbf", ".poni"))

    box = {}
    w = SolveWorker(img, LAM, PX * 1e6)
    w.done.connect(lambda r: box.update(r=r))
    w.run()
    res = box["r"]
    cx, cy, D_px = res["cx"], res["cy"], res["D_px"]

    tf = fit_tilt(img, cx, cy, D_px, phase_qs("CeO2"), LAM, PX, PX,
                  sfull=1, nsec=36, mass_rel=0.10, bump=1.5)
    ai_plan = mk_ai(D_px * PX, (cy + 0.5) * PX, (cx + 0.5) * PX, 0, 0)
    if tf:
        ai_tilt = mk_ai(tf["dist"], tf["poni1"], tf["poni2"],
                        tf["rot1"], tf["rot2"])
    else:
        ai_tilt = None

    s_plan, _ = prof_sharpness(ai_plan, img, qs_use)
    s_tilt, _ = (prof_sharpness(ai_tilt, img, qs_use)
                 if ai_tilt else (float("nan"), None))
    s_gt, _ = prof_sharpness(gt, img, qs_use)

    g2 = gt.getFit2D()
    gtc = np.hypot(g2["centerX"], g2["centerY"]) * 0  # placeholder
    if ai_tilt:
        t2 = ai_tilt.getFit2D()
        gap = float(np.hypot(t2["centerX"] - g2["centerX"],
                             t2["centerY"] - g2["centerY"]))
        rot_s = (f"rot=({np.degrees(tf['rot1'])*1000:+.1f},"
                 f"{np.degrees(tf['rot2'])*1000:+.1f})mdeg "
                 f"imp={tf['imp']*100:.0f}%")
    else:
        gap = float("nan")
        rot_s = "fit_tilt None"
    print(f"ceria_{d}: Σ锐度 平面={s_plan:.3e}  +tilt={s_tilt:.3e}  "
          f"GT={s_gt:.3e}  | tilt-Fit2D中心 vs GT-Fit2D中心 "
          f"{gap:.1f}px | {rot_s}", flush=True)
