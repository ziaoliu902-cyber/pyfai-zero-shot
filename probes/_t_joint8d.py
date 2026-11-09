# -*- coding: utf-8 -*-
"""_t_joint8d.py:B13 生产链正解——mean-frame 种子(2026-11-08)。

溯源结论(nc_audit_v0_1 ③/starve_guard):alumina 弱衍射静态装置
逐帧盲解次优盆丛生(per-frame 相鉴 16/52,逐帧 MAD 19mm 级),
116.79 是 mean-frame 管线产物。本探针验证联合标定的正确前置:

  8 帧 → 平均(mean-frame)→ 一次盲解(锁相)→ 该解做**全部帧**
  fit_tilt 种子 → 每帧控制点 → joint_solve_frames。

判决:Δ(GT)= −0.6mm 级(GT 种子水准 −0.588);每帧独立 D 的
MAD ≈ 0.08mm(控制点口径,对照 B13 51 帧 0.084)。
副产品:该链每帧 ~15s(免逐帧盲解 ~5min),52 帧 ~15 分钟。
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
from torchfai.phases import phase_qs

PX1, PX2 = 103.358e-6, 102.530e-6
LAM = 12.398 / 17.5
GT = 116.860
QS = phase_qs("Al2O3")
AL = os.path.join(ROOT, "data", "real", "alumina")
frames_all = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                    and f.endswith(".edf"))
sample = [frames_all[i] for i in (0, 6, 12, 18, 24, 30, 36, 42)]

# ---- 1) mean-frame + 一次盲解 ----
acc = None
imgs = {}
for f in sample:
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    imgs[f] = img
    acc = img.copy() if acc is None else acc + img
mean_img = acc / len(sample)
print("mean-frame(8 帧)盲解(锁相)…", flush=True)
r = sc.solve_image(mean_img, PX1, lam=LAM, phases=["Al2O3"])
assert r is not None, "mean-frame 盲解无解"
s = r["s"]
D_seed = r["sol"][2] * s * PX1 * 1e3
print(f"  mean-frame 种子:D={D_seed:.3f}mm (Δ{D_seed-GT:+.3f}) "
      f"m={r['sol'][3]} rms={r['sol'][4]:.3f}px")

# ---- 2) 每帧控制点(同一种子) ----
cpts, Ds = [], []
cx_n = r["sol"][0] * s
cy_n = r["sol"][1] * s
D_n = r["sol"][2] * s
for f in sample:
    c = sc.cpts_of_frame(imgs[f], cx_n, cy_n, D_n, PX1, PX2, LAM, QS)
    cpts.append(c)
    if c:
        from poni_out import fit_tilt as _ft
        Ds.append(_ft(imgs[f], cx_n, cy_n, D_n, QS, LAM, PX1, PX2,
                      sfull=1, nsec=36, mass_rel=0.10)["dist"] * 1e3)
Ds = np.array(Ds)
print(f"\n每帧 fit_tilt(同种子):成功 {len(Ds)}/8 "
      f"D 中位={np.median(Ds):.3f} MAD="
      f"{np.median(np.abs(Ds-np.median(Ds))):.3f}mm")

# ---- 3) 联合 ----
res = sc.joint_solve_frames(cpts, PX1, PX2, boot=6)
assert res is not None
Dj = res["dist"] * 1e3
print(f"\n联合解:D={Dj:.3f}mm (GT {GT}, Δ{Dj-GT:+.3f}) "
      f"rot=({np.degrees(res['rot1']):+.3f},{np.degrees(res['rot2']):+.3f})°"
      f" rms={res['rms']*1e3:.2f}mrad npts={res['npts']}")
print(f"  σ_boot={res['sigma_boot']:.3f}mm(6×70%)")

ok1 = abs(Dj - GT) < 1.0
ok2 = (res["sigma_boot"] or 1) < 0.2
print(f"\n判决:Δ {'✓' if ok1 else '✗'}(<1mm,GT 种子水准); "
      f"σ_boot {'✓' if ok2 else '✗'}(<0.2mm)")
print("总判:" + ("✓ mean-frame 种子链达标" if ok1 and ok2 else "✗"))
sys.exit(0 if ok1 and ok2 else 1)
