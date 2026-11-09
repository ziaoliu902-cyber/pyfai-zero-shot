# -*- coding: utf-8 -*-
"""_t_joint8h.py:B 路线 spline-mean 种子复活判决(2026-11-08)。

链:8 帧 → b_route_mean_seed(流式 mean+画布圆解+spline fit_tilt)
→ 每帧 cpts(spline 前向)→ joint_solve_frames(spline)。
判决:种子 Δ<0.5(B 路线复活);联合 Δ<1.0;帧独立 D MAD<0.15。
对照臂:同链无 spline(联合应复现 Δ≈−0.59 系统差→验证口径效应)。
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
frames_all = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                    and f.endswith(".edf"))
sample = [frames_all[i] for i in (0, 6, 12, 18, 24, 30, 36, 42)]
imgs = []
for f in sample:
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    imgs.append(np.where(np.isfinite(img), img, 0.0))

# ---- 1) B 路线 mean 种子(spline) ----
print("B 路线 mean 种子(8 帧,spline)…", flush=True)
seed = sc.b_route_mean_seed(iter(imgs), PX1, PX2, LAM, "Al2O3",
                            spline=SPL)
assert seed is not None, "种子失败"
D_seed = seed["dist"] * 1e3
print(f"  种子:D={D_seed:.3f}mm (Δ{D_seed-GT:+.3f}) "
      f"rot=({np.degrees(seed['rot1']):+.4f},{np.degrees(seed['rot2']):+.4f})°"
      f" rms5={seed['rms5']*1e3:.2f}mrad n={seed['n']}")
ok_seed = abs(D_seed - GT) < 0.5

cx_n = seed["poni2"] / PX2 - 0.5
cy_n = seed["poni1"] / PX1 - 0.5
D_n = seed["dist"] / PX1

# ---- 2) 每帧控制点 + 联合(spline 口径) ----
fwd_spl = make_pyfai_fwd(PX1, PX2, SPL)
for tag, use_spl in (("spline 全链", True), ("无 spline 对照", False)):
    cpts, Df = [], []
    fw = fwd_spl if use_spl else None
    sp = SPL if use_spl else None
    for f, img in zip(sample, imgs):
        c = sc.cpts_of_frame(img, cx_n, cy_n, D_n, PX1, PX2, LAM, QS,
                             fwd=fw)
        cpts.append(c)
        if c:
            Df.append(c["D"])
    Df = np.array(Df)
    res = sc.joint_solve_frames(cpts, PX1, PX2, boot=6, spline=sp)
    assert res is not None
    Dj = res["dist"] * 1e3
    mad = float(np.median(np.abs(Df - np.median(Df)))) if len(Df) else 9
    print(f"\n[{tag}] 每帧 fit_tilt:n={len(Df)}/8 "
          f"D中位={np.median(Df):.3f} MAD={mad:.3f}mm")
    print(f"  联合:D={Dj:.3f}mm (Δ{Dj-GT:+.3f}) "
          f"rot=({np.degrees(res['rot1']):+.4f},"
          f"{np.degrees(res['rot2']):+.4f})° rms={res['rms']*1e3:.2f}mrad"
          f" σ_boot={res['sigma_boot']:.3f}mm npts={res['npts']}")
    if use_spl:
        ok_j = abs(Dj - GT) < 1.0
        ok_mad = mad < 0.15
    else:
        ok_j, ok_mad = None, None

print(f"\n判决:种子 Δ {'✓' if ok_seed else '✗'}(<0.5); "
      f"联合(spline)Δ {'✓' if ok_j else '✗'}(<1.0); "
      f"帧 MAD {'✓' if ok_mad else '✗'}(<0.15)")
allok = ok_seed and ok_j and ok_mad
print("总判:" + ("✓ B 路线种子复活,弱衍射域打通" if allok else "✗"))
sys.exit(0 if allok else 1)
