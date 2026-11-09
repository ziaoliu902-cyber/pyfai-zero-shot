# -*- coding: utf-8 -*-
"""_b13_joint.py:B13 多帧全局拟合(2026-11-07)。

问题:alumina 逐帧解 MAD 0.39mm(锁相);均值帧(像素平均)已达
Δ−0.07mm,但装置漂移时会糊。控制点级联合拟合 = 全局 (D,rot1,rot2)
+ 每帧 (poni1_i,poni2_i):静态与缓漂移都可用。

协议(统计层判决,种子=GT 附近):
  1) 52 帧各跑 fit_tilt(GT 种子,data_out)拿控制点(d1,d2,tth,w,qidx)
  2) 联合 LSQ:3 全局参 + 2×52 帧参,残差=tth 模型−目标(×w)
  3) 对照:逐帧 fit_tilt D 的 MAD;bootstrap 抽帧(70%,×20)的 σ(D_joint)
判决:σ(D_joint) ≈ MAD/√N(√51 收敛)?|D_joint−GT|?

GT:al2o3_00_max_51_frames.poni D=116.860mm λ=0.7084571Å
px1=103.358µm px2=102.530µm(各向异性)。
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
from poni_out import fit_tilt, make_pyfai_fwd
from torchfai.phases import phase_qs

LAM = 12.398 / 17.5
PX1, PX2 = 103.358e-6, 102.530e-6
GT_D = 116.860
GT_P1, GT_P2 = 52.95653e-3, 54.73342e-3     # poni1/poni2(m)
GT_R1, GT_R2 = 0.01582112, 0.00940420
AL = os.path.join(ROOT, "data", "real", "alumina")
QS = phase_qs("Al2O3")

frames = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                and f.endswith(".edf"))
print(f"B13 联合拟合:{len(frames)} 帧,锁相 Al2O3,λ={LAM:.7f}Å")

# ---- 1) 逐帧控制点(GT 种子) ----
CACHE = os.path.join(ROOT, "probes", "_b13_cpts.npz")
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    Ds = z["Ds"]; npts = z["npts"]
    C = [z[f"c{i}"] for i in range(len(Ds))]
    print("(控制点缓存命中)")
else:
    Ds, npts, C = [], [], []
    cx0 = GT_P2 / PX2 - 0.5        # fit_tilt 的 cx/cy 是画布索引(含 +0.5 中心约定)
    cy0 = GT_P1 / PX1 - 0.5
    D_px0 = GT_D * 1e-3 / PX1
    for k, f in enumerate(frames):
        img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
        img = np.where(np.isfinite(img), img, 0.0)
        do = {}
        tf = fit_tilt(img, cx0, cy0, D_px0, QS, LAM, PX1, PX2,
                      sfull=1, nsec=36, mass_rel=0.10, data_out=do)
        if tf is None or len(do.get("d1", [])) < 20:
            Ds.append(np.nan); npts.append(0); C.append(None)
        else:
            Ds.append(tf["dist"] * 1e3); npts.append(len(do["d1"]))
            C.append(dict(d1=do["d1"], d2=do["d2"], tth=do["tth"],
                          w=do["w"], p1=tf["poni1"], p2=tf["poni2"],
                          r1=tf["rot1"], r2=tf["rot2"]))
        if (k + 1) % 10 == 0:
            print(f"  ... {k+1}/{len(frames)}", flush=True)
    Ds = np.array(Ds); npts = np.array(npts)
    np.savez(CACHE, Ds=Ds, npts=npts,
             **{f"c{i}": C[i] for i in range(len(C))})

ok = np.isfinite(Ds)
print(f"逐帧 fit_tilt:成功 {ok.sum()}/{len(frames)}  "
      f"D 中位={np.nanmedian(Ds):.3f}mm (GT {GT_D}) "
      f"MAD={np.nanmedian(np.abs(Ds[ok]-np.nanmedian(Ds))):.3f}mm")

# ---- 2) 联合 LSQ:全局(dist,rot1,rot2)+每帧(poni1,poni2) ----
from scipy.optimize import least_squares

live = [i for i in range(len(C)) if C[i] is not None]
fwd = make_pyfai_fwd(PX1, PX2, None)
D1 = np.concatenate([C[i]["d1"] for i in live])
D2 = np.concatenate([C[i]["d2"] for i in live])
TT = np.concatenate([C[i]["tth"] for i in live])
WW = np.concatenate([C[i]["w"] for i in live])
FR = np.concatenate([[k] * len(C[i]["d1"]) for k, i in enumerate(live)])
WW = WW / WW.mean()
nf = len(live)


def resid(p):
    dist = p[0]; r1 = p[1]; r2 = p[2]
    p1 = p[3 + 2 * FR]; p2 = p[4 + 2 * FR]
    tm = fwd(dist, p1, p2, r1, r2, D1, D2)
    return (tm - TT) * WW


p0 = np.concatenate([[GT_D * 1e-3, GT_R1, GT_R2],
                     np.array([[C[i]["p1"], C[i]["p2"]] for i in live]).ravel()])
print(f"\n联合拟合:控制点 {len(TT)} / 帧 {nf} / 参数 {len(p0)}(3 全局+"
      f"{2*nf} 帧)")
fit = least_squares(resid, p0, method="lm", max_nfev=4000)
D_joint = fit.x[0] * 1e3
rms_j = float(np.sqrt(np.mean(resid(fit.x) ** 2))) * 1e3   # mrad
print(f"联合解:D={D_joint:.3f}mm (GT {GT_D}, Δ{D_joint-GT_D:+.3f}) "
      f"rot=({np.degrees(fit.x[1]):+.3f}°,{np.degrees(fit.x[2]):+.3f}°) "
      f"rms={rms_j:.2f}mrad")

# ---- 3) bootstrap 抽帧 70%×20 → σ(D_joint) ----
rng = np.random.default_rng(3)
Ds_b = []
for b in range(20):
    sel = rng.choice(nf, size=max(8, int(nf * 0.7)), replace=False)
    selS = set(sel)
    mask = np.isin(FR, list(selS))
    D1b, D2b, TTb, WWb, FRb = D1[mask], D2[mask], TT[mask], WW[mask], FR[mask]
    remap = {v: k for k, v in enumerate(sorted(selS))}
    FRb = np.array([remap[v] for v in FRb])

    def rb(pp):
        p1 = pp[3 + 2 * FRb]; p2 = pp[4 + 2 * FRb]
        return (fwd(pp[0], p1, p2, pp[1], pp[2], D1b, D2b) - TTb) * WWb

    p0b = np.concatenate([[fit.x[0], fit.x[1], fit.x[2]],
                          np.array([[C[live[v]]["p1"], C[live[v]]["p2"]]
                                    for v in sorted(selS)]).ravel()])
    fb = least_squares(rb, p0b, method="lm", max_nfev=2500)
    Ds_b.append(fb.x[0] * 1e3)
    if (b + 1) % 5 == 0:
        print(f"  bootstrap {b+1}/20", flush=True)

Ds_b = np.array(Ds_b)
sig = float(np.std(Ds_b))
print(f"\nbootstrap σ(D_joint)={sig:.3f}mm  (20×70% 抽帧)")
print(f"逐帧 MAD/√n = {np.nanmedian(np.abs(Ds[ok]-np.nanmedian(Ds)))/np.sqrt(ok.sum()):.3f}mm")
print("\n判决:σ(D_joint) ≤ MAD/√n×1.5 且 |ΔGT|<0.3mm → 联合拟合达标"
      "(√N 收敛);漂移场景另测。")
