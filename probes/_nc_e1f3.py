# -*- coding: utf-8 -*-
r"""_nc_e1f3.py — E1f3 A 臂：E9M/Eiger4M/P2M 走 E1f2 新路径（SolveWorker
+ hybrid_ai V3），中心 vs GT-Fit2D + Σ锐度 vs GT 几何。

预期：E9M（rot~0.9° 真倾角）大胜；Eiger4M（rot 11.6/−5.9mrad）改善；
P2M 负对照——小角域倾角门必须守住（判词不得差于 0.010° 量级水位）。
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


CASES = [
    ("Eiger9M", "Eiger9M.h5", "Eiger9M.poni"),
    ("Eiger4M", "Eiger4M.edf", "Eiger4M.poni"),
    ("Pilatus2M", "Pilatus2M.cbf", "Pilatus2M.poni"),
]

for tag, img_f, poni_f in CASES:
    os.chdir(ZOO)
    p = pyFAI.load(os.path.join(ZOO, poni_f))
    px_um = float(p.detector.pixel1) * 1e6
    lam = (p.wavelength * 1e10) if p.wavelength else 1.0
    gt = pyFAI.load(os.path.join(ZOO, poni_f))
    g2 = gt.getFit2D()
    img = load_any(os.path.join(ZOO, img_f))
    img = np.where(img >= 4e9, 0.0, img)

    box = {}
    w = SolveWorker(img, lam, px_um)
    w.done.connect(lambda r: box.update(r=r))
    w.run()
    res = box.get("r")
    if not res:
        print(f"== {tag} == SOLVE FAIL", flush=True)
        continue
    ai_v3 = hybrid_ai(res, px_um)
    f2 = ai_v3.getFit2D()
    dc = float(np.hypot(f2["centerX"] - g2["centerX"],
                        f2["centerY"] - g2["centerY"]))
    dD = ai_v3.dist * 1000 - gt.dist * 1000
    tf = res.get("tf")
    qs = phase_qs(res["ph"])
    qs_s = [q for q in qs if 1.0 < q < 3.6][:6]
    s_ours = sigma_sharp(ai_v3, img, qs_s)
    s_gt = sigma_sharp(gt, img, qs_s)
    print(f"== {tag} ==  ph={res['ph']} tilt_on={res.get('tilt_on')} "
          f"tf_native={res.get('tf_native')} "
          + (f"imp={tf['imp']*100:.0f}%" if tf else "tf=None"), flush=True)
    print(f"  V3几何: 中心vsGT-Fit2D={dc:6.2f}px  ΔD={dD:+6.2f}mm  "
          f"Σ锐度 ours/gt = {s_ours:.3e} / {s_gt:.3e} "
          f"({s_ours/max(s_gt,1e-9)*100:.0f}%)", flush=True)
    print(f"  判词参考: check_std 走 hybrid_ai 同源（见向导）", flush=True)
    print(flush=True)
