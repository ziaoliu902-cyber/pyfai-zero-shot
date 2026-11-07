# -*- coding: utf-8 -*-
"""A3/A4 端到端离屏测试(2026-11-07):真解 → σ → 报告 PDF/md。

ceria_200 真帧:SolveWorker 同步跑(离屏),UncWorker(b_tilt=6) 同步,
报告双格式落盘断言;wizard _unc_done 文案断言。
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from PyQt5.QtWidgets import QApplication, QMessageBox

import gui_solver
import gui_wizard

QMessageBox.critical = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

app = QApplication([])
w = gui_solver.Win()

F = ROOT + r"\data\real\MX_ceria\ceria_200_1_0001.cbf"
w._load(F)
assert w.img is not None and w.img_path == F
print("载图 ✓", w.img.shape)

# ---- 真解(同步) ----
out = {}
wk = gui_solver.SolveWorker(w.img, None, 172.0)
wk.done.connect(lambda r: out.__setitem__("res", r))
wk.fail.connect(lambda tb: out.__setitem__("fail", tb))
wk.run()
assert "fail" not in out, out.get("fail")
res = out["res"]
assert res is not None, "ceria_200 未解出(不应发生)"
print(f"解 ✓ ph={res['ph']} D={res['D_px']*172e-3:.1f}mm m={res['sol'][3]}")

# ---- σ(同步,b_tilt=6 只验管道) ----
w.res = res
w.on_done(res)
u = {}
uw = gui_solver.UncWorker(w.img, res, 172.0, b_tilt=6)
uw.done.connect(lambda r: u.__setitem__("u", r))
uw.fail.connect(lambda tb: u.__setitem__("fail", tb))
uw.run()
assert "fail" not in u, u.get("fail")
w.on_unc_done(u["u"])
txt = w.txt.toPlainText()
assert "σ(D)" in txt, txt[-300:]
print("σ(D) ✓", [ln for ln in txt.splitlines() if "σ(D)" in ln][-1])

# ---- 报告 PDF + md ----
pdf_p = ROOT + r"\runs\_rpt_test.pdf"
md_p = ROOT + r"\runs\_rpt_test.md"
saved = {"n": 0}
gui_solver.QFileDialog.getSaveFileName = staticmethod(
    lambda *a, **k: (pdf_p if saved["n"] == 0 else md_p, ""))
saved["n"] += 0
w.export_report()
saved["n"] += 1
assert os.path.exists(pdf_p) and os.path.getsize(pdf_p) > 10000, "PDF 未生成"
print(f"PDF ✓ {os.path.getsize(pdf_p)/1024:.0f}KB")
gui_solver.QFileDialog.getSaveFileName = staticmethod(
    lambda *a, **k: (md_p, ""))
w.export_report()
assert os.path.exists(md_p) and "σ(D)" in open(md_p, encoding="utf-8").read()
print("md ✓")

# ---- wizard σ 文案 ----
gui_wizard.tr  # import 校验
wz = gui_wizard.Wizard()
wz.show()
wz.p3._unc_done(dict(bs=(0.5, 197.8, 199.0, 8), tb=None, tf=None))
assert "σ(D)" in wz.p3.out.toPlainText()
print("wizard σ 文案 ✓")

os.remove(pdf_p); os.remove(md_p)
print("全部断言通过")
