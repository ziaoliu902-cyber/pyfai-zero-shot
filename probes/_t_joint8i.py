# -*- coding: utf-8 -*-
"""_t_joint8i.py:b_route_mean_seed 失败定位(2026-11-08)。"""
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc
from poni_out import fit_tilt, make_pyfai_fwd
from torchfai.phases import phase_qs

PX1, PX2 = 103.358e-6, 102.530e-6
LAM = 0.7084811
QS = phase_qs("Al2O3")
SPL = os.path.join(ROOT, "data", "real", "alumina", "distorsion_2x2.spline")
AL = os.path.join(ROOT, "data", "real", "alumina")
frames_all = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                    and f.endswith(".edf"))
sample = [frames_all[i] for i in (0, 6, 12, 18, 24, 30, 36, 42)]
acc = None
for f in sample:
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    acc = img if acc is None else acc + img
mean_img = acc / len(sample)

r = sc.solve_image(mean_img, (PX1 + PX2) / 2, lam=LAM, phases=["Al2O3"],
                   solve_kw=dict(cascade=True, auto_center=False,
                                 refine_2d=False))
print("solve_image(mean):", "None" if r is None else
      f"D={r['sol'][2]*r['s']*(PX1+PX2)/2*1e3:.3f}mm s={r['s']}")
if r is None:
    sys.exit(1)
s = r["s"]
for tag, fw in (("自研前向", None),
                ("spline 前向", make_pyfai_fwd(PX1, PX2, SPL))):
    tf = fit_tilt(r["canvas"], r["sol"][0], r["sol"][1], r["sol"][2],
                  QS, LAM, PX1 * s, PX2 * s, sfull=s, fwd=fw, nsec=36,
                  mass_rel=0.10, bump=1.5)
    if tf is None:
        print(f"fit_tilt({tag}): None")
    else:
        print(f"fit_tilt({tag}): D={tf['dist']*1e3:.3f}mm "
              f"rms5={tf['rms5']*1e3:.2f}mrad npts={tf['npts']} "
              f"rot=({np.degrees(tf['rot1']):+.4f},"
              f"{np.degrees(tf['rot2']):+.4f})°")
print("results/alumina_mean.poni 存在:",
      os.path.exists(ROOT + r"\probes\results\alumina_mean.poni"))
