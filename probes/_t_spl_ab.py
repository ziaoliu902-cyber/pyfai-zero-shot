# -*- coding: utf-8 -*-
"""_t_spl_ab.py：A/B 对照——改动前(_gui_solver_pre) vs 改动后(gui_solver)
同图同参跑 SolveWorker+check_std，逐字段对比证明 T-SPL 接线零回归。"""
import sys

sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
from PyQt5.QtWidgets import QApplication  # noqa
app = QApplication([])
import i18n
i18n.set_lang("zh")

IMG = r"C:\Users\liu\Desktop\衍射测试图\ceria_400_1_0001.cbf"


def run(modname):
    mod = __import__(modname)
    img = mod.load_any(IMG)
    meta = mod.sniff_meta(IMG)
    lam = float("{:g}".format(meta["lam"]))
    box = {}
    w = mod.SolveWorker(img, lam, 172.0)
    w.done.connect(lambda r: box.update(r=r))
    w.fail.connect(lambda tb: (_ for _ in ()).throw(RuntimeError(tb)))
    w.run()
    res = box["r"]
    c = mod.check_std(img, res, 172.0)
    return dict(ph=res["ph"], lam=res["lam"], D_mm=res["D_px"] * 172e-3,
                cx=res["cx"], cy=res["cy"], native=res["native"],
                tilt_on=res["tilt_on"], center_src=res["center_src"],
                ring_rms=res["ring_rms"], n=c["n"], dev_max=c["dev_max"],
                verdict=c["verdict"])


a = run("_gui_solver_pre")
b = run("gui_solver")
keys = list(a.keys())
print(f"{'字段':<12}{'改动前':>24}{'改动后':>24}  一致")
all_ok = True
for k in keys:
    va, vb = a[k], b[k]
    if isinstance(va, float):
        same = abs(va - vb) < 1e-9
    else:
        same = va == vb
    all_ok &= same
    print(f"{k:<12}{str(va):>24}{str(vb):>24}  {'✓' if same else '✗'}")
print("\nA/B 判决：" + ("✅ 逐字段一致——T-SPL 接线零回归" if all_ok else "❌ 有差异"))
sys.exit(0 if all_ok else 1)
