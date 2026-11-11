# -*- coding: utf-8 -*-
"""_t_p4r_smoke.py:④红灯行动建议冒烟(2026-11-08)。

mock CheckWorker 失败结果(dev_max 大)→ _advice 应给:
①λ未填时:填 λ 建议+按钮,回②动作把 λ̂ 填入框;
②环少/谱弱:多帧建议;③λ已填+谱强:通用建议。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probes"))
os.chdir(ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication
app = QApplication([])

import gui_wizard as gw

w = gw.Wizard()
w.p1._load_many([r"data\real\MX_ceria\ceria_150_1_0001.cbf"])  # 单张
w.p2.e_px.setText("172")
# res mock:m=4(环少),ps=0.30(谱弱),λ=0.6889(联立值)
w.res = dict(ph="CeO2", lam=0.6889, D_px=800.0, cx=1000, cy=1000,
             sol=[0, 0, 0, 4, 0.5, 0, 0.30], canvas=None, s=1.0)
w.res_j = None

# ① λ 空 → 填 λ 建议在场 + 按钮 + 动作填值
w.p2.e_lam.setText("")
c = dict(lines=["mock"], n=6, dev_max=0.30, verdict="FAIL")
w.p4._done(c)
txt = w.p4.out.toPlainText()
assert "回第②步填已知 λ" in txt, txt
assert w.p4.act_row is not None
w.p4._go_fill_lam()
assert w.p2.e_lam.text() == "0.6889", w.p2.e_lam.text()
print("① λ 空 → 填λ建议 + 一键回②填λ̂ ✓")

# ② 谱弱 + λ 空 → 同时给多帧建议
w.p2.e_lam.setText("")
w.p4.out.setPlainText("")
w.p4._done(c)
txt = w.p4.out.toPlainText()
assert "多拍几张" in txt, txt
print("② 环少/谱弱 → 多帧建议 ✓")

# ③ λ 已填 → 无填λ建议,给通用条
w.p2.e_lam.setText("0.6889")
w.p4.out.setPlainText("")
w.res["sol"][3] = 12          # 环多
w.res["sol"][6] = 0.80        # 谱强
w.p4._done(c)
txt = w.p4.out.toPlainText()
assert "填已知 λ" not in txt
assert "多拍几张" not in txt
assert "近似距离" in txt
print("③ λ已填+谱强 → 通用建议(近似距离) ✓")

print("\n总判:✓ ④红灯行动建议冒烟全通")
