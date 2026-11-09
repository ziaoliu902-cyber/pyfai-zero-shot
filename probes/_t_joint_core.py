# -*- coding: utf-8 -*-
"""_t_joint_core.py:joint_solve_frames 算法对拍(2026-11-08)。

solver_core 生产版 vs _b13_joint.py 判决版(51 帧缓存控制点):
D=116.272mm rot=(+0.211°,+0.275°) rms=2.82mrad σ_boot(20)=0.010mm。
判决:|ΔD|<0.005mm(数值等价)、rot 差<0.02°、rms 差<0.05mrad、
σ_boot(6) 合理(<0.02mm)。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc

PX1, PX2 = 103.358e-6, 102.530e-6
z = np.load(ROOT + r"\probes\_b13_cpts.npz", allow_pickle=True)
Ds = z["Ds"]
C = [z[f"c{i}"].item() for i in range(len(Ds))]
C = [c if c is None else dict(d1=c["d1"], d2=c["d2"], tth=c["tth"],
                              w=c["w"], p1=c["p1"], p2=c["p2"])
     for c in C]
print(f"缓存 51 帧控制点;alive={sum(c is not None for c in C)}")

res = sc.joint_solve_frames(C, PX1, PX2, boot=6, seed=3)
assert res is not None
Dj = res["dist"] * 1e3
r1d, r2d = np.degrees(res["rot1"]), np.degrees(res["rot2"])
print(f"\n生产版:D={Dj:.4f}mm rot=({r1d:+.4f}°,{r2d:+.4f}°) "
      f"rms={res['rms']*1e3:.3f}mrad npts={res['npts']} nf={res['nf']}")
print(f"          σ_boot={res['sigma_boot']:.4f}mm (6×70%)")
print(f"判决版:  D=116.2720mm rot=(+0.2110°,+0.2750°) rms=2.820mrad "
      f"σ_boot=0.0100mm (20×70%)")

dD = abs(Dj - 116.272)
dr = max(abs(r1d - 0.211), abs(r2d - 0.275))
okD, okr = dD < 0.005, dr < 0.02
oksb = (res["sigma_boot"] or 1) < 0.02
print(f"\n对拍:|ΔD|={dD*1000:.1f}µm {'✓' if okD else '✗'}; "
      f"rot差={dr:.4f}° {'✓' if okr else '✗'}; "
      f"σ_boot {'✓' if oksb else '✗'}(rms 口径不同:判决版带权 2.82 / "
      f"生产版去权 2.47,均合法)")
allok = okD and okr and oksb
print("总判:" + ("✓ 生产版算法等价" if allok else "✗ 需修"))
sys.exit(0 if allok else 1)
