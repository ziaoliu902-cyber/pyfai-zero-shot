# -*- coding: utf-8 -*-
r"""_t_spl_gui.py：T-SPL 产品级全链路（Win 真方法，离屏）。

覆盖需求 F1/F2/F3/F4 的 GUI 侧：
  apply_corr（monkeypatch 文件对话框）→ 状态可见 → 求解走校正图
  → export_poni 写 SplineFile → undo_corr 还原原图。
e2e 数值验收（|ΔD|<0.1mm）由 _t_spl_e2e.py 负责，本脚本只验接线。
用法：python -X utf8 probes\_t_spl_gui.py
"""
import os
import sys
import tempfile

sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication, QFileDialog  # noqa
app = QApplication([])
import i18n
i18n.set_lang("zh")

import numpy as np
import pyFAI

import gui_solver
from gui_solver import Win

IMG = r"D:\pyfai-project\data\real\MX_ceria\ceria_150_1_0001.cbf"
SPL = r"D:\pyfai-project\data\real\silx_zoo\frelon.spline"
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'✓' if cond else '✗'}] {name}")


win = Win()
win._load(IMG)
raw = win.img.copy()
print("1) 初始态（F6）")
check("spline=None", win.spline is None)
check("img_raw=None", win.img_raw is None)
check("撤销按钮隐藏", win.b_sp_off.isHidden())
check("sp_lab 空", win.sp_lab.text() == "")

print("2) apply_corr（F1/F2/F5）")
QFileDialog.getOpenFileName = (lambda *a, **k: (SPL, ""))
win.apply_corr()
check("spline 记录", os.path.normcase(win.spline or "") == os.path.normcase(SPL))
check("原图备份", win.img_raw is not None and win.img_raw.shape == raw.shape
      and np.array_equal(win.img_raw, raw))
m = 16
inner = np.asarray(win.img[m:-m, m:-m])
check("图像已变（校正生效）", not np.array_equal(win.img, raw))
check(f"边带已置零（{m}px）", win.img[:m].max() == 0 and win.img[:, :m].max() == 0)
check("中部数据保留", float(np.median(inner)) != 0.0)
check("撤销按钮可见", not win.b_sp_off.isHidden())
check("状态标签含文件名", os.path.basename(SPL) in win.sp_lab.text())
check("旧 res 已作废", win.res is None and not win.b_poni.isEnabled())

print("3) 校正图参与求解（F3）——SolveWorker 直吃 win.img")
box = {}
w = gui_solver.SolveWorker(win.img, 0.972386, 172.0)
w.done.connect(lambda r: box.update(r=r))
w.fail.connect(lambda tb: (_ for _ in ()).throw(RuntimeError(tb)))
w.run()
res = box["r"]
check("解出", res is not None)
win.res = res
win.b_poni.setEnabled(True)

print("4) export_poni 写 SplineFile（F4）")
tmp = os.path.join(tempfile.gettempdir(), "_t_spl_gui.poni")
QFileDialog.getSaveFileName = (lambda *a, **k: (tmp, ""))
win.export_poni()
d = pyFAI.load(tmp).detector
back = getattr(d, "splinefile", None) or getattr(d, "splineFile", None)
check("poni 回读 SplineFile 一致",
      bool(back) and os.path.normcase(back) == os.path.normcase(SPL))

print("5) undo_corr（F2）")
win.undo_corr()
check("回到原图", np.array_equal(win.img, raw))
check("spline 清空", win.spline is None and win.img_raw is None)
check("撤销按钮再隐藏", win.b_sp_off.isHidden())
tmp2 = os.path.join(tempfile.gettempdir(), "_t_spl_gui2.poni")
if os.path.exists(tmp2):
    os.remove(tmp2)
QFileDialog.getSaveFileName = (lambda *a, **k: (tmp2, ""))
win.export_poni()
check("撤销后旧解已作废，导出被阻止", not os.path.exists(tmp2))
win.res = res                     # 模拟在原图上重解
win.export_poni()
d2 = pyFAI.load(tmp2).detector
back2 = getattr(d2, "splinefile", None) or getattr(d2, "splineFile", None)
check("未校正导出的 poni 不带 SplineFile", not back2)

print("6) 换图清校正态（F6）")
QFileDialog.getOpenFileName = (lambda *a, **k: (SPL, ""))
win.apply_corr()
win._load(IMG)
check("换图后 spline=None", win.spline is None and win.img_raw is None)

print("\n判决：" + ("✅ GUI 全链路通过" if ok_all else "❌ 有失败项"))
sys.exit(0 if ok_all else 1)
