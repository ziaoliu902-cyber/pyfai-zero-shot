# -*- coding: utf-8 -*-
"""离屏冒烟：真图跑一次 SolveWorker，确认 σ_D / σ_rot 能出来并渲染到面板。"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import numpy as np

import gui_solver as G
from PyQt5.QtWidgets import QApplication

app = QApplication([])
w = G.Win()

img = G.load_any(os.path.join(ROOT, "data", "real",
                              "MX_ceria/ceria_150_1_0001.cbf"))
print("image", img.shape)

res = []
worker = G.SolveWorker(img, 0.972386, 172.0)
worker.done.connect(lambda r: res.append(r))
worker.fail.connect(lambda tb: res.append(("FAIL", tb)))
worker.run()

r = res[0]
if isinstance(r, tuple):
    print(r[1])
    sys.exit(1)
print("keys:", sorted(r.keys()))
print("sigma_D_mm:", round(r.get("sD_mm", 0), 3),
      " sigma_rot(mrad):", round(r.get("s_r1", 0) * 1e3, 2),
      round(r.get("s_r2", 0) * 1e3, 2),
      " tilt_on:", r.get("tilt_on"))
w.img = img
w.on_done(r)
print("---- 面板 ----")
print(w.txt.toPlainText())

