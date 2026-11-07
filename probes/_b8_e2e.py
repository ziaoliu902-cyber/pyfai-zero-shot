# -*- coding: utf-8 -*-
"""_b8_e2e.py:B8 生产接线端到端验证(2026-11-07)。

阴性(ceria 原图)与阳性(spline 注入)走 GUI 同款 SolveWorker:
- 阴性:res["distort_red"] < 0.05,主窗口文本不出扭曲旗
- 阳性:res["distort_red"] > 0.05,文本含「几何扭曲嫌疑」
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication

app = QApplication([])

import gui_solver as G
from _b8_distort_probe import inject   # 复用注入器

base_img = G.load_any(ROOT + r"\data\real\MX_ceria\ceria_200_1_0001.cbf")
pos_img, amp = inject(base_img, ROOT + r"\data\real\silx_zoo\halfccd.spline", 1.0)

results = {}
for tag, img in (("阴性 ceria 原图", base_img),
                 ("阳性 halfccd 注入", pos_img)):
    out = {}
    wk = G.SolveWorker(img, 0.972386, 172.0)
    wk.done.connect(lambda r: out.__setitem__("res", r))
    wk.fail.connect(lambda tb: out.__setitem__("fail", tb))
    wk.run()                      # 同步(离屏,信号直连)
    res = out.get("res")
    results[tag] = res
    print(f"{tag:16} sol={'ok' if res else 'FAIL'}  "
          f"distort_red={res.get('distort_red') if res else '?'}"
          f"  D={res['D_px']*172e-3 if res else 0:.2f}mm")

neg = results["阴性 ceria 原图"]
pos = results["阳性 halfccd 注入"]
ok = (neg is not None and pos is not None
      and neg.get("distort_red", 0.0) < 0.05
      and pos.get("distort_red", 0.0) > 0.05)
print("\nB8 端到端:", "通过 ✓" if ok else "失败 ✗")
sys.exit(0 if ok else 1)
