# -*- coding: utf-8 -*-
"""_t_lamu_wiz.py:wizard 页面 2 单位切换离屏验证(2026-11-07)。"""
import sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from PyQt5.QtWidgets import QApplication
app = QApplication([])

from gui_wizard import PageParam

p = PageParam()
ok = True


def chk(tag, got, want, tol=1e-9):
    global ok
    good = (abs(got - want) <= tol) if isinstance(want, float) else got == want
    print(f"  [{'✓' if good else '✗'}] {tag}: {got!r}"
          + ("" if good else f" (期望 {want!r})"))
    ok = ok and good


p.e_lam.setText("0.972386"); p.cb_lamu.setCurrentIndex(0)
chk("Å 直读", p._lam_A(), 0.972386)
p.cb_lamu.setCurrentIndex(1)
chk("切 keV 后 _lam_A", p._lam_A(), 0.972386)
p.cb_lamu.setCurrentIndex(0)
chk("切回 Å", p._lam_A(), 0.972386)
p.e_lam.setText(""); p.cb_lamu.setCurrentIndex(1); p.e_lam.setText("72.1")
chk("72.1 keV", p._lam_A(), 12.39842 / 72.1)
p.e_lam.setText("")
chk("空=联立解", p._lam_A(), None)
print("\n" + ("wizard 通过 ✓" if ok else "wizard 失败 ✗"))
sys.exit(0 if ok else 1)
