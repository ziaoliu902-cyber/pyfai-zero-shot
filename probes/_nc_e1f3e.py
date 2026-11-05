# -*- coding: utf-8 -*-
r"""_nc_e1f3e.py — E1f3e A/B：SolveWorker 原生优先 tf（非 native 帧移植）。

判决点：Eiger4M/v2 中心 9.32/5.62px → 预期 ~0.3/0.8px（E1f3d zoo 同款）；
ceria_400 走画布兜底不变（native 拟合 None，弱环）；P2M 负对照不动。
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

ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
CERIA = os.path.join(ROOT, "data", "real", "MX_ceria")
CASES = [
    ("Eiger4M", os.path.join(ZOO, "Eiger4M.edf"),
     os.path.join(ZOO, "Eiger4M.poni")),
    ("Eiger4M_v2", os.path.join(ZOO, "Eiger4M.edf"),
     os.path.join(ZOO, "Eiger4M_v2.poni")),
    ("ceria_400", os.path.join(CERIA, "ceria_400_1_0001.cbf"),
     os.path.join(CERIA, "ceria_400_1_0001.poni")),
    ("Pilatus2M", os.path.join(ZOO, "Pilatus2M.cbf"),
     os.path.join(ZOO, "Pilatus2M.poni")),
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
    if not res:
        print(f"== {tag} == SOLVE FAIL", flush=True)
        continue
    ai3 = hybrid_ai(res, px_um)
    f3 = ai3.getFit2D()
    dc = float(np.hypot(f3["centerX"] - g2["centerX"],
                        f3["centerY"] - g2["centerY"]))
    tf = res.get("tf")
    print(f"== {tag} == ph={res['ph']} native={res.get('native')} "
          f"tilt_on={res.get('tilt_on')} tf_native={res.get('tf_native')} "
          + (f"imp={tf['imp']*100:.0f}% " if tf else "tf=None ")
          + f"Δc(V3)={dc:.2f}px ΔD={ai3.dist*1000-gt.dist*1000:+.2f}mm",
          flush=True)
