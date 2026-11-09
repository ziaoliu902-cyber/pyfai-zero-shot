# -*- coding: utf-8 -*-
"""_t_gui_joint_smoke.py:batch_gui 联合功能冒烟(2026-11-08)。

offscreen 构造 BatchWorker,假控制点池验证 _run_joint 分组/联合/
返回结构;不弹窗不跑图。
"""
import os
import sys

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from batch_gui import BatchWorker, BatchDialog
from PyQt5.QtWidgets import QApplication

app = QApplication([])

PX = 103.358
LAM = 0.7085


def fake_cpts(cx_px=505.0, cy_px=500.0, n=40, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(150, 480, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    d2 = cx_px + r * np.cos(ph)
    d1 = cy_px + r * np.sin(ph)
    tth = np.arctan(r * PX * 1e-6 / 0.1168) + rng.normal(0, 2e-4, n)
    return dict(d1=d1, d2=d2, tth=tth, w=np.ones(n),
                p1=(cy_px + 0.5) * PX * 1e-6,
                p2=(cx_px + 0.5) * PX * 1e-6, rms5=2e-3, D=116.0)


w = BatchWorker(["f0.edf", "f1.edf", "f2.edf", "f3.edf", "f4.edf"], PX,
                LAM, want_joint=True)
pool = [(0, "Al2O3", PX, LAM, fake_cpts(seed=0)),
        (1, "Al2O3", PX, LAM, fake_cpts(seed=1)),
        (2, "Quartz", PX, LAM, fake_cpts(seed=2)),     # 异相 → 应被分组淘汰
        (3, "Al2O3", 99.0, LAM, fake_cpts(seed=3)),    # 异 px → 同上
        (4, "Al2O3", PX, LAM, fake_cpts(seed=4))]
j = w._run_joint(pool)
assert j is not None, "联合 None"
print(f"冒烟:_run_joint → nf={j['nf']}(应 3,Quartz/99µm 已剔) "
      f"D={j['dist']*1e3:.3f}mm files={len(j['files'])}")
assert j["nf"] == 3 and len(j["files"]) == 3
assert "sigma_boot" in j and "p1s" in j and "px_um" in j

dlg = BatchDialog(px_um=PX, lam=str(LAM))
assert dlg.cb_joint.isChecked(), "联合 checkbox 默认应开"
print("冒烟:BatchDialog 构造 ok,cb_joint 默认 ✓,b_jponi 存在 "
      f"{'✓' if hasattr(dlg, 'b_jponi') else '✗'}")
print("总判:✓ GUI 联合链冒烟通过")

# ---- B 路线 mean 模式冒烟(2026-11-08):结构验证 ----
dlg2 = BatchDialog(px_um=103.358, lam="0.7085")
assert not dlg2.cb_mean.isChecked(), "mean 默认应关"
dlg2.cb_mean.setCheckState(2)
assert dlg2.cb_phase.isEnabled(), "mean 勾选后相下拉应启用"
dlg2.cb_phase.setCurrentText("Al2O3")
w2 = BatchWorker(["a.edf", "b.edf", "c.edf"], 103.358, 0.7085,
                 want_joint=True, want_mean=True, phase_hint="Al2O3")
assert w2.want_mean and w2.phase_hint == "Al2O3"
assert hasattr(w2, "_run_mean_mode") and hasattr(w2, "_sniff_spline")
print("冒烟:mean 模式控件/worker 参数/方法存在 ✓")
print("总判(mean):✓")
