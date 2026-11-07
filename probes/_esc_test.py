# -*- coding: utf-8 -*-
"""环不足自动调档·离屏逻辑测试(不弹真对话框)。

验证:爬档顺序 高→极高、cb_sens 同步、爬完恢复原档、on_fail 中途恢复。
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from PyQt5.QtWidgets import QApplication

import solver_core as sc
import gui_solver

# 桩掉模态对话框(offscreen 下 exec_ 会卡死)
gui_solver.QMessageBox.critical = staticmethod(lambda *a, **k: None)
gui_solver.QMessageBox.warning = staticmethod(lambda *a, **k: None)

app = QApplication([])
w = gui_solver.Win()

calls = []
w._retry_solve = lambda: calls.append(sc._RING_SENS)   # 不走 QTimer

# ---- 场景 1:标准档起步,连败爬升 高→极高→恢复 ----
assert sc._RING_SENS == 0 and w.cb_sens.currentIndex() == 0
w._esc_from, w._esc_next = 0, 1
w.on_done(None)                       # → 试高
assert (sc._RING_SENS, w.cb_sens.currentIndex()) == (1, 1)
w.on_done(None)                       # → 试极高
assert (sc._RING_SENS, w.cb_sens.currentIndex()) == (2, 2)
w.on_done(None)                       # → 爬完恢复标准
assert (sc._RING_SENS, w.cb_sens.currentIndex()) == (0, 0)
assert w._esc_from is None and w._esc_next is None
assert calls == [1, 2], calls
print("场景1 爬升/恢复 ✓", calls)

# ---- 场景 2:高档起步,只在极高试一次 ----
sc.set_ring_sensitivity(1); w.cb_sens.setCurrentIndex(1)
w._esc_from, w._esc_next = 1, 2
w.on_done(None)
assert (sc._RING_SENS, w.cb_sens.currentIndex()) == (2, 2)
w.on_done(None)
assert (sc._RING_SENS, w.cb_sens.currentIndex()) == (1, 1)
assert w._esc_from is None
print("场景2 高档起步 ✓")

# ---- 场景 3:on_fail 爬升途中异常 → 恢复原档并清状态 ----
sc.set_ring_sensitivity(0); w.cb_sens.setCurrentIndex(0)
w._esc_from, w._esc_next = 0, 1
w.on_fail("boom")                     # QMessageBox.critical 也会弹!
print("场景3 状态:", sc._RING_SENS, w.cb_sens.currentIndex(),
      w._esc_from, w._esc_next)

# ---- 场景 4:状态干净时 on_fail 不动档位 ----
sc.set_ring_sensitivity(2); w.cb_sens.setCurrentIndex(2)
w.on_fail("boom2")
assert sc._RING_SENS == 2 and w._esc_from is None
print("场景4 ✓")

sc.set_ring_sensitivity(0); w.cb_sens.setCurrentIndex(0)
print("全部断言通过")
