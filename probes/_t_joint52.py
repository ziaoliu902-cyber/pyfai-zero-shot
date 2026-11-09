# -*- coding: utf-8 -*-
"""_t_joint52.py:B 路线种子 + 52 帧联合终验(2026-11-08)。

joint8j 复刻成功:52 帧 mean 圆解 118.25(+1.39)→ fit_tilt(画布+
spline)116.7256(Δ−0.134,=alumina_mean.poni)。8 帧 mean 信噪
不足(圆解崩 72)→ **B 路线必须全帧 mean**。
终验(生产函数链):b_route_mean_seed(52)→ 每帧 cpts(spline)
→ joint_solve_frames(spline)。判决:种子 Δ<0.5;联合 Δ<0.8;
帧 MAD<0.15;σ_boot<0.1。
"""
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
from poni_out import make_pyfai_fwd
from torchfai.phases import phase_qs

PX1, PX2 = 103.358e-6, 102.530e-6
LAM = 0.7084811
GT = 116.860
QS = phase_qs("Al2O3")
SPL = os.path.join(ROOT, "data", "real", "alumina", "distorsion_2x2.spline")
AL = os.path.join(ROOT, "data", "real", "alumina")
frames = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                and f.endswith(".edf"))
print(f"B 路线生产函数链终验:{len(frames)} 帧全量")

# ---- 1) mean 种子(函数版,isfinite 清洗) ----
def gen():
    for f in frames:
        img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
        yield np.where(np.isfinite(img), img, 0.0)

print("mean 种子(52 帧,画布+spline fit_tilt)…", flush=True)
seed = sc.b_route_mean_seed(gen(), PX1, PX2, LAM, "Al2O3", spline=SPL)
assert seed is not None, "种子失败"
Ds_ = seed["dist"] * 1e3
print(f"  种子:D={Ds_:.4f}mm (Δ{Ds_-GT:+.4f}) "
      f"rot=({np.degrees(seed['rot1']):+.4f},"
      f"{np.degrees(seed['rot2']):+.4f})° n={seed['n']}")
ok_seed = abs(Ds_ - GT) < 0.5
print(f"  种子判决:{'✓' if ok_seed else '✗'}(目标 116.7256)")

cx_n = seed["poni2"] / PX2 - 0.5
cy_n = seed["poni1"] / PX1 - 0.5
D_n = seed["dist"] / PX1

# ---- 2) 每帧控制点(自研 fwd) ----
cpts, Df = [], []
for k, f in enumerate(frames):
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    c = sc.cpts_of_frame(img, cx_n, cy_n, D_n, PX1, PX2, LAM, QS)
    cpts.append(c)
    if c:
        Df.append(c["D"])
    if (k + 1) % 10 == 0:
        print(f"  … {k+1}/{len(frames)}", flush=True)
Df = np.array(Df)
mad = float(np.median(np.abs(Df - np.median(Df))))
print(f"\n每帧 fit_tilt(自研 fwd):{len(Df)}/{len(frames)} "
      f"D中位={np.median(Df):.4f} MAD={mad:.4f}mm")

# ---- 3) 联合(spline) ----
res = sc.joint_solve_frames(cpts, PX1, PX2, boot=8)
assert res is not None
Dj = res["dist"] * 1e3
print(f"\n联合:D={Dj:.4f}mm (GT {GT}, Δ{Dj-GT:+.4f}) "
      f"rot=({np.degrees(res['rot1']):+.4f},{np.degrees(res['rot2']):+.4f})°"
      f" rms={res['rms']*1e3:.2f}mrad npts={res['npts']} "
      f"σ_boot={res['sigma_boot']:.4f}mm")

ok_j = abs(Dj - GT) < 0.8
ok_mad = mad < 0.15
ok_sb = res["sigma_boot"] < 0.1
print(f"\n判决:种子 {'✓' if ok_seed else '✗'};联合 Δ {'✓' if ok_j else '✗'}"
      f"(<0.8);帧 MAD {'✓' if ok_mad else '✗'}(<0.15); "
      f"σ_boot {'✓' if ok_sb else '✗'}(<0.1)")
allok = ok_seed and ok_j and ok_mad and ok_sb
print("总判:" + ("✓ 弱衍射域全链打通(mean 种子→spline 联合)"
                if allok else "✗"))
sys.exit(0 if allok else 1)
