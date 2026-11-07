# -*- coding: utf-8 -*-
"""A2 批量模式·离屏测试(2026-11-07):3 帧 ceria 真跑 + CSV + poni + 表格。

漂移曲线(≥5 帧)走 BatchWorker 层验证数据装配,弹窗本身不 exec。
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from PyQt5.QtWidgets import QApplication, QMessageBox

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.critical = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import gui_solver
import batch_gui

app = QApplication([])

FILES = [ROOT + rf"\data\real\MX_ceria\ceria_{d}_1_0001.cbf"
         for d in (150, 200, 250)]

dlg = batch_gui.BatchDialog(px_um=172.0, lam="0.972386")
dlg._set_files(FILES)
assert len(dlg.files) == 3

# ---- BatchWorker 同步直跑(离屏无事件循环,信号照样 emit 到本线程) ----
wk = batch_gui.BatchWorker(dlg.files, 172.0, 0.972386, want_sigma=True)
wk._auto_px = True
rows = []
wk.row.connect(lambda i, d: rows.append(d))
wk.run()
assert len(rows) == 3
for d, gt in zip(rows, (150.0, 200.0, 250.0)):
    assert d.get("res"), d["status"]
    assert abs(d["D_mm"] - gt) < 12, (d["D_mm"], gt)
    assert d["sD"] != "" and d["sD"] < 3
    print(f"  {os.path.basename(d['file'])}: D={d['D_mm']}mm "
          f"(GT {gt}) m={d['m']} σ_D={d['sD']}")
print("批量解+σ ✓")

# ---- 表格装配(on_row)+CSV ----
dlg.table.setRowCount(3)
dlg.rows = [None] * 3
for i, d in enumerate(rows):
    dlg.on_row(i, d)
assert dlg.table.item(0, 1).text() == "CeO2"
csv_p = ROOT + r"\runs\_batch_test.csv"
batch_gui.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (csv_p, ""))
dlg.rows = rows
dlg.export_csv()
txt = open(csv_p, encoding="utf-8-sig").read()
assert "ceria_150" in txt and "CeO2" in txt
print("CSV ✓")

# ---- poni 批导出(写 <图名>_fit.poni,用临时副本避免污染数据目录) ----
import shutil
import tempfile
tmp = tempfile.mkdtemp()
copies = []
for f in FILES:
    c = os.path.join(tmp, os.path.basename(f))
    shutil.copy(f, c)
    copies.append(c)
dlg.rows = []
wk2 = batch_gui.BatchWorker(copies, 172.0, 0.972386, want_sigma=False)
wk2._auto_px = True
r2 = []
wk2.row.connect(lambda i, d: r2.append(d))
wk2.run()
dlg.rows = r2
dlg.export_poni_all()
ponis = [x for x in os.listdir(tmp) if x.endswith("_fit.poni")]
assert len(ponis) == 3, ponis
print("poni ×3 ✓")

# ---- 漂移数据装配(不弹窗):≥5 帧门槛逻辑 ----
pts = [(i, r["D_mm"], r["sD"] or 0.0)
       for i, r in enumerate(dlg.rows) if r and r.get("res")]
assert len(pts) == 3
print("漂移装配(3 帧样本,<5 不开按钮) ✓")

os.remove(csv_p)
shutil.rmtree(tmp)
print("全部断言通过")
