# -*- coding: utf-8 -*-
"""组合精度测试：经典 → (可选 r2d 原生精化) → fit_tilt，比较 D 与倾角。

对比三种路径：
  A. 经典 + fit_tilt（当前生产）
  B. 经典 + r2d + fit_tilt（分层组合）
输出：每帧 ΔD、Δrot1/2（对 GT），以及合矢量统计。
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
import pyFAI

from solver_core import to_canvas, solve_cascade, refine_geometry_2d
from torchfai.phases import phase_qs
from poni_out import fit_tilt

LAM = 0.972386
PX = 172e-6
Q = phase_qs("CeO2")[:8]


def run(tag, use_r2d):
    dD, rot = [], []
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(
            ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = p.dist * 1000
        canvas, s, _, _ = to_canvas(img)
        pxm = PX * s
        sol = solve_cascade(canvas, LAM, pxm, Q, rms_floor=1.8)
        cx, cy, D = sol[0], sol[1], sol[2]
        if use_r2d:
            o = refine_geometry_2d(img, cx * s, cy * s, D * s, Q, LAM, n_iter=3)
            if o is not None:
                cx, cy, D = o[0] / s, o[1] / s, o[2] / s
        tf = fit_tilt(canvas, cx, cy, D, Q, LAM, pxm, pxm, sfull=s,
                      nsec=36, mass_rel=0.10, bump=1.5)
        if tf is None:
            dD.append(np.nan); rot.append((np.nan, np.nan)); continue
        dD.append(tf["dist"] * 1000 - gt)
        rot.append(((tf["rot1"] - p.rot1) * 1e3, (tf["rot2"] - p.rot2) * 1e3))
    dD = np.array(dD); rot = np.array(rot)
    mag = np.hypot(rot[:, 0], rot[:, 1])
    print(f"{tag}: |ΔD| 平均 {np.nanmean(np.abs(dD)):.2f} 最差 "
          f"{np.nanmax(np.abs(dD)):.2f} mm； 倾角 |Δrot| 平均 "
          f"{np.nanmean(mag):.2f} mrad = {np.degrees(np.nanmean(mag)/1e3):.4f}°，"
          f"最大 {np.nanmax(mag):.2f} mrad")
    print("   逐帧 ΔD:", " ".join(f"{x:+.2f}" for x in dD))
    return dD, rot


print("A = 经典 + fit_tilt（生产现状）")
run("A", False)
print("\nB = 经典 + r2d + fit_tilt（组合）")
run("B", True)
