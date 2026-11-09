# -*- coding: utf-8 -*-
"""_t_fwdspl2.py:同帧同种子,自研 fwd vs spline fwd 的 fit_tilt
对照(2026-11-08)。分辩:每帧 −1.9 是 spline fwd 口径问题还是
单帧织构峰位问题。"""
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from poni_out import fit_tilt, make_pyfai_fwd
from torchfai.phases import phase_qs

PX1, PX2 = 103.358e-6, 102.530e-6
LAM = 0.7084811
QS = phase_qs("Al2O3")
SPL = os.path.join(ROOT, "data", "real", "alumina",
                   "distorsion_2x2.spline")
GT = 116.860

# mean 种子(joint52: D=116.7277 poni≈GT)
seed_D = 116.7277e-3
seed_p1, seed_p2 = 0.0522, 0.0514     # 约值;用 GT poni 精值更好
import re
kv = dict(re.findall(r"^(\w+):\s*([-\d.eE+]+)$", open(os.path.join(
    ROOT, "data", "real", "alumina", "al2o3_00_max_51_frames.poni")
    ).read(), re.M))
seed_p1, seed_p2 = float(kv["Poni1"]), float(kv["Poni2"])
cx_n = seed_p2 / PX2 - 0.5
cy_n = seed_p1 / PX1 - 0.5
D_n = seed_D / PX1

f_native = make_pyfai_fwd(PX1, PX2, None)
f_spl = make_pyfai_fwd(PX1, PX2, SPL)
AL = os.path.join(ROOT, "data", "real", "alumina")
frames = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                and f.endswith(".edf"))
for f in frames[:3]:
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    print(f"\n{f}(种子 D={seed_D*1e3:.3f}):")
    for tag, fw in (("自研", f_native), ("spline", f_spl)):
        do = {}
        tf = fit_tilt(img, cx_n, cy_n, D_n, QS, LAM, PX1, PX2, sfull=1,
                      fwd=fw, nsec=36, mass_rel=0.10, bump=1.5,
                      data_out=do)
        if tf is None:
            print(f"  [{tag}] None")
            continue
        print(f"  [{tag}] D={tf['dist']*1e3:.4f} (Δ{tf['dist']*1e3-GT:+.4f})"
              f" rot=({np.degrees(tf['rot1']):+.4f},"
              f"{np.degrees(tf['rot2']):+.4f})° npts={tf['npts']} "
              f"rms5={tf['rms5']*1e3:.2f}mrad")
