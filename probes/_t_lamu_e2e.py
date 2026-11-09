# -*- coding: utf-8 -*-
"""_t_lamu_e2e.py:能量输入端到端(2026-11-07)。

P2M CdTe CeO2(GT 800mm):keV 模式填 72.1 → run_solve 全管线 →
解 D 与 Å 模式等价(λ=12.39842/72.1 vs 惯例 12.398/72.1 差 <1e-4,
D 差 <0.1mm)。判据:|D−800| < 0.5mm 且与 keV/Å 两模式互差 <0.15mm。
"""
import sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QEventLoop, QTimer

app = QApplication([])
from gui_solver import Win

w = Win()
w._load(ROOT + r"\data\real\Pilatus2MCdTe_ID15_CeO2_72100eV_800mm_0000.cbf")


def solve_once(unit, val):
    w.cb_lamu.setCurrentIndex(0 if unit == "A" else 1)
    w.e_lam.setText(val)
    loop = QEventLoop()
    w.worker = None
    QTimer.singleShot(50, w.run_solve)

    def done(res):
        loop.quit()
        done.res = res
    # run_solve 里 worker 是新建的,延迟挂钩
    def hook():
        if w.worker is not None:
            w.worker.done.connect(done)
            return True
        return False
    t = QTimer(); t.setInterval(20)
    t.timeout.connect(lambda: hook() and t.stop())
    t.start()
    loop.exec_()
    return done.res


r_kev = solve_once("keV", "72.1")
D_kev = r_kev["D_px"] * 172e-3
print(f"keV 模式(72.1 keV): ph={r_kev['ph']} D={D_kev:.3f}mm "
      f"m={r_kev['sol'][3]} rms={r_kev['sol'][4]:.2f}")

r_a = solve_once("A", "0.1719614")     # 12.39842/72.1
D_a = r_a["D_px"] * 172e-3
print(f"Å 模式(0.1719614): ph={r_a['ph']} D={D_a:.3f}mm "
      f"m={r_a['sol'][3]} rms={r_a['sol'][4]:.2f}")

d1, d2 = abs(D_kev - 800.0), abs(D_a - 800.0)
print(f"\n判据:|D−800|<0.5 → keV:{d1:.3f} Å:{d2:.3f};"
      f" 互差 {abs(D_kev - D_a):.3f}mm < 0.15")
ok = d1 < 0.5 and d2 < 0.5 and abs(D_kev - D_a) < 0.15
print("端到端 " + ("通过 ✓" if ok else "失败 ✗"))
sys.exit(0 if ok else 1)
