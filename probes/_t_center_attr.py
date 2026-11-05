# -*- coding: utf-8 -*-
r"""_t_center_attr.py：ceria_150 自检 ⚠0.166° 归因——中心 vs D 尺度。

变体实验（同一 SolveWorker 解）：
  V0 原解                     → dev_max 基线
  V1 中心换共心真值（D 不动）  → 中心份额
  V2 D 按自检单参 k 修正（中心不动）→ 尺度份额
  V3 双修                     → 残余地板（环模型/找峰极限）
"""
import sys

sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
from PyQt5.QtWidgets import QApplication  # noqa
app = QApplication([])
import i18n
i18n.set_lang("zh")

import numpy as np
from gui_solver import load_any, sniff_meta, SolveWorker, check_std
from solver_core import concentric_center
from torchfai.phases import phase_qs

IMG = r"D:\pyfai-project\data\real\MX_ceria\ceria_150_1_0001.cbf"
img = load_any(IMG)
lam = float("{:g}".format(sniff_meta(IMG)["lam"]))

box = {}
w = SolveWorker(img, lam, 172.0)
w.done.connect(lambda r: box.update(r=r))
w.fail.connect(lambda tb: (_ for _ in ()).throw(RuntimeError(tb)))
w.run()
res = box["r"]

ref = concentric_center(img, (res["cx"], res["cy"]), res["D_px"],
                        phase_qs(res["ph"]), lam)
print(f"解: 中心=({res['cx']:.2f},{res['cy']:.2f}) D={res['D_px']*172e-3:.2f}mm"
      f"  src={res['center_src']}")
print(f"ref=({ref['cx']:.2f},{ref['cy']:.2f}) σ=({ref['sx']:.3f},{ref['sy']:.3f})"
      f"  我们距ref={np.hypot(res['cx']-ref['cx'], res['cy']-ref['cy']):.2f}px")


def dev(res_v):
    c = check_std(img, res_v, 172.0)
    return c["n"], c["dev_max"], c["verdict"]


def cp(cx, cy, D_px):
    r2 = dict(res)
    r2["cx"], r2["cy"], r2["D_px"] = cx, cy, D_px
    return r2


n0, d0, v0 = dev(res)
print(f"\nV0 原解                : n={n0} dev_max={d0:.3f}°  {v0}")

n1, d1, v1 = dev(cp(ref["cx"], ref["cy"], res["D_px"]))
print(f"V1 中心→ref            : n={n1} dev_max={d1:.3f}°  {v1}")

# V2：用 V0 的单参 k（check_std 内部拟合，重算一遍同式）
c0 = check_std(img, res, 172.0)
print("\n（V0 各环 Δ2θ 明细）")
for ln in c0["lines"]:
    if "vs" in ln:
        print("  ", ln.strip())

# k 从明细重新拟合：Δ2θ = −k·tan2θ
import re
tp, tm = [], []
for ln in c0["lines"]:
    m = re.search(r"([\d.]+)°\s+vs\s+([\d.]+)°\s+Δ([+-][\d.]+)°", ln)
    if m:
        tp.append(float(m.group(1)))
        tm.append(float(m.group(2)))
tp, tm = np.radians(tp), np.radians(tm)
y = tm - tp
tx = np.tan(tp)
k = -float(tx @ y / (tx @ tx))
print(f"单参尺度 k=δD/D={k:+.5f}（D 修正 {k*res['D_px']*172e-3:+.3f}mm）")

n2, d2, v2 = dev(cp(res["cx"], res["cy"], res["D_px"] * (1 - k)))
print(f"V2 D×(1−k) 尺度修正     : n={n2} dev_max={d2:.3f}°  {v2}")

n3, d3, v3 = dev(cp(ref["cx"], ref["cy"], res["D_px"] * (1 - k)))
print(f"V3 中心→ref + 尺度修正  : n={n3} dev_max={d3:.3f}°  {v3}")
