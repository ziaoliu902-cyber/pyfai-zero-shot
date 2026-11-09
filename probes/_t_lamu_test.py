# -*- coding: utf-8 -*-
"""_t_lamu_test.py:λ/能量单位切换 + σ(D) λ 传播 离屏回归(2026-11-07)。

覆盖:主窗口/批量对话框的单位换算往返、_lam_A 三态(空/Å/keV)、
λ 传播项(给定 λ → 1e-3·D;联立解 → 0)、语法级 import。
"""
import sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from PyQt5.QtWidgets import QApplication
app = QApplication([])

from gui_solver import Win as SolverWindow
from batch_gui import BatchDialog

w = SolverWindow()
ok = True


def chk(tag, got, want, tol=0.0):
    global ok
    good = (abs(got - want) <= tol) if isinstance(want, float) else got == want
    print(f"  [{'✓' if good else '✗'}] {tag}: {got!r}"
          + ("" if good else f" (期望 {want!r})"))
    ok = ok and good


print("1) 主窗口:Å 直读")
w.e_lam.setText("0.972386"); w.cb_lamu.setCurrentIndex(0)
chk("_lam_A Å", w._lam_A(), 0.972386, 1e-9)
chk("_lam_text", w._lam_text(), "0.972386")
chk("_lam_prop_mm(200.19)", w._lam_prop_mm(200.19), 0.20019, 1e-4)

print("2) 切 keV:框内换算 + 往返一致")
w.cb_lamu.setCurrentIndex(1)
e_keV = float(w.e_lam.text())
chk("框内 keV 值", e_keV, 12.39842 / 0.972386, 1e-6)
chk("_lam_A(keV 模式)", w._lam_A(), 0.972386, 1e-9)
chk("_lam_text(仍 Å 口径)", w._lam_text(), "0.972386")
w.cb_lamu.setCurrentIndex(0)
chk("切回 Å 后 _lam_A", w._lam_A(), 0.972386, 1e-9)

print("3) 空框 = 联立解")
w.e_lam.setText("")
chk("_lam_A 空", w._lam_A(), None)
chk("_lam_text 空", w._lam_text(), "")
chk("_lam_prop_mm 空→0", w._lam_prop_mm(200.19), 0.0)

print("4) 典型能量 72.1 keV(P2M 域;先切单位再填数=真实操作流)")
w.e_lam.setText(""); w.cb_lamu.setCurrentIndex(1)
w.e_lam.setText("72.1")
chk("_lam_A(72.1keV)", w._lam_A(), 12.39842 / 72.1, 1e-9)
chk("_lam_prop_mm(800)", w._lam_prop_mm(800.0), 0.8, 1e-9)

print("5) 批量对话框同款")
w.e_lam.setText(""); w.cb_lamu.setCurrentIndex(0)
bd = BatchDialog(None, px_um=172.0, lam="0.5")
bd.e_lam.setText("0.5"); bd.cb_lamu.setCurrentIndex(0)
chk("batch Å", bd._lam_A(), 0.5, 1e-12)
bd.cb_lamu.setCurrentIndex(1)
chk("batch keV 往返", bd._lam_A(), 0.5, 1e-9)
bd.e_lam.setText("")
chk("batch 空", bd._lam_A(), None)

print("\n" + ("全部通过 ✓" if ok else "存在失败 ✗"))
sys.exit(0 if ok else 1)
