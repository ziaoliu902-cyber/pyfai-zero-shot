# -*- coding: utf-8 -*-
"""_offer_recovery 两分支离屏测试:选「自动调」/「填波长」。"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from PyQt5.QtWidgets import QApplication

import solver_core as sc
import gui_solver

gui_solver.QMessageBox.critical = staticmethod(lambda *a, **k: None)
gui_solver.QMessageBox.warning = staticmethod(lambda *a, **k: None)

app = QApplication([])
w = gui_solver.Win()
calls = []
w._retry_solve = lambda: calls.append(("solve", sc._RING_SENS,
                                       w.e_lam.text()))

# ---- 分支 A:选「没有 λ,自动调灵敏度」----
_click = {}


class _FakeBox:
    def __init__(self, *a, **k):
        self._btns = {}

    def setWindowTitle(self, *a):
        pass

    def setIcon(self, *a):
        pass

    def setText(self, *a):
        pass

    def addButton(self, label, role):
        import types
        b = types.SimpleNamespace(label=label)
        self._btns[label] = b
        return b

    def exec_(self):
        _click["box"] = self

    def clickedButton(self):
        return self._btns[_click["pick"]]


gui_solver.QMessageBox = _FakeBox  # 只在本测试范围内
gui_solver.QMessageBox.Question = 1
gui_solver.QMessageBox.AcceptRole = 0
gui_solver.QMessageBox.RejectRole = 1

w.e_lam.clear()
_click["pick"] = "没有 λ,自动调灵敏度"
w._offer_recovery()
assert w._esc_from == 0 and w._esc_next == 2, (w._esc_from, w._esc_next)
assert sc._RING_SENS == 1 and w.cb_sens.currentIndex() == 1
assert calls and calls[-1][1] == 1
print("分支A 自动调 ✓ → 档位", sc._RING_SENS)

# ---- 分支 B:选「填波长 λ」,输入 0.5 ----
sc.set_ring_sensitivity(0); w.cb_sens.setCurrentIndex(0)
w._esc_from = w._esc_next = None


class _FakeInput:
    def __init__(self, *a, **k):
        pass


def _get_text(parent, title, label, text=""):
    return ("0.5", True)


# _offer_recovery 用的是 gui_solver.QInputDialog.getText
class _QID:
    @staticmethod
    def getText(*a, **k):
        return ("0.5", True)


gui_solver.QInputDialog = _QID
w.e_lam.clear()
_click["pick"] = "填波长 λ"
w._offer_recovery()
assert w.e_lam.text() == "0.5", w.e_lam.text()
assert calls[-1] == ("solve", 0, "0.5"), calls[-1]
assert w._esc_from is None
print("分支B 填波长 ✓ → e_lam =", w.e_lam.text(), "λ 已随重解生效")

# ---- 分支 C:输入无效波长 → 不重试、不弹档 ----
calls.clear()
w.e_lam.clear()


class _QID2:
    @staticmethod
    def getText(*a, **k):
        return ("abc", True)


gui_solver.QInputDialog = _QID2
gui_solver.QMessageBox.warning = staticmethod(lambda *a, **k: None)
_click["pick"] = "填波长 λ"
w._offer_recovery()
assert not calls and w.e_lam.text() == "" and w._esc_from is None
print("分支C 无效波长拒收 ✓")

sc.set_ring_sensitivity(0)
print("全部通过")
