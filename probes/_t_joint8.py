# -*- coding: utf-8 -*-
"""_t_joint8.py:B13 生产化测试——solver_core 联合标定两层(2026-11-08)。

判决(生产同路径:盲解画布解做种子,不用 GT):
  alumina 8 帧抽样锁相 Al2O3 → solve_image(种子)→ cpts_of_frame
  → joint_solve_frames:|ΔD_joint vs GT| 与逐帧中位同量级(≤2mm),
  σ_boot 合理(≤0.1mm 量级),失败帧不炸。
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

LAM = 12.398 / 17.5
PX1, PX2 = 103.358e-6, 102.530e-6
GT = 116.860
AL = os.path.join(ROOT, "data", "real", "alumina")
QS = phase_qs("Al2O3")

frames_all = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                    and f.endswith(".edf"))
sample = [frames_all[i] for i in (0, 6, 12, 18, 24, 30, 36, 42)]
print(f"联合标定两层测试:8 帧锁相(λ={LAM:.6f}, px={PX1*1e6:.3f}µm)")

cpts, Ds, fails = [], [], []
SEEDS = os.path.join(ROOT, "probes", "_t_joint8_seeds.npz")
seeddata = {}
if os.path.exists(SEEDS):
    z = np.load(SEEDS, allow_pickle=True)
    seeddata = {k: z[k].item() for k in z.files}
    print("(盲解种子缓存命中)")
for k, f in enumerate(sample):
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    if f in seeddata:
        sd = seeddata[f]
        r, s = sd["r"], sd["s"]
        Ds.append(sd["D_mm"])
    else:
        r = sc.solve_image(img, PX1, lam=LAM, phases=["Al2O3"])
        if r is None:
            print(f"  [{k+1}/8] {f}: 盲解无解(记失败帧)")
            cpts.append(None)
            fails.append(f)
            continue
        s = r["s"]
        Ds.append(r["sol"][2] * s * PX1 * 1e3)
        seeddata[f] = dict(r=r, s=s, D_mm=Ds[-1])
    cx_n = r["sol"][0] * s          # 画布→原生(线性换算)
    cy_n = r["sol"][1] * s
    D_n = r["sol"][2] * s
    c = sc.cpts_of_frame(img, cx_n, cy_n, D_n, PX1, PX2, LAM, QS)
    tag = f"{len(c['d1'])}pt" if c else "cpts失败"
    print(f"  [{k+1}/8] {f}: D={Ds[-1]:.2f}mm(Δ{Ds[-1]-GT:+.2f}) "
          f"种子→{tag}", flush=True)
    cpts.append(c)

np.savez(SEEDS, **{k: v for k, v in seeddata.items()})

ok = [d for d in Ds if np.isfinite(d)]
print(f"\n逐帧盲解:n={len(ok)} 中位={np.median(ok):.2f}mm "
      f"(Δ{np.median(ok)-GT:+.2f}) 失败帧 {len(fails)}")

# ---- win 扫描:盲解种子下峰窗宽对控制点质量的影响 ----
print("\nwin 扫描(fit_tilt 峰窗):")
for win_ in (6.0, 8.0, 10.0, 12.0):
    cc, nn, rr = [], 0, []
    for k, f in enumerate(sample):
        if cpts[k] is None and f not in seeddata:
            cc.append(None)
            continue
        img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
        img = np.where(np.isfinite(img), img, 0.0)
        sd = seeddata[f]
        r_, s_ = sd["r"], sd["s"]
        c = sc.cpts_of_frame(img, r_["sol"][0] * s_, r_["sol"][1] * s_,
                             r_["sol"][2] * s_, PX1, PX2, LAM, QS,
                             win=win_)
        cc.append(c)
        if c:
            nn += len(c["d1"])
            rr.append(c["rms5"])
    res_w = sc.joint_solve_frames(cc, PX1, PX2, boot=6)
    if res_w is None:
        print(f"  win={win_}: 联合失败")
        continue
    Dw = res_w["dist"] * 1e3
    print(f"  win={win_}: npts={nn} 帧rms中位="
          f"{np.median(rr)*1e3:.2f}mrad → D={Dw:.3f} (Δ{Dw-GT:+.3f}) "
          f"σ={res_w['sigma_boot']:.3f}")
cpts = cc     # 用最优 win(下一轮联合用 12.0 档,与扫描末一致)

res = sc.joint_solve_frames(cpts, PX1, PX2, boot=6)
assert res is not None, "联合解 None"
print(f"\n联合解:D={res['dist']*1e3:.3f}mm (GT {GT}, "
      f"Δ{res['dist']*1e3-GT:+.3f})  rot=({np.degrees(res['rot1']):+.3f},"
      f"{np.degrees(res['rot2']):+.3f})°  rms={res['rms']*1e3:.2f}mrad "
      f"npts={res['npts']} nf={res['nf']}")
print(f"  σ_boot={res['sigma_boot']:.3f}mm(6×70%)"
      if res.get("sigma_boot") else "  σ_boot:不足")
print(f"  每帧中心 p1 p2 首尾:({res['p1s'][0]*1e3:.2f},"
      f"{res['p2s'][0]*1e3:.2f})…({res['p1s'][-1]*1e3:.2f},"
      f"{res['p2s'][-1]*1e3:.2f})mm")

dj = abs(res["dist"] * 1e3 - GT)
ok1 = dj < 2.0
ok2 = (res.get("sigma_boot") or 1.0) < 0.2
print(f"\n判决:Δ={dj:.3f}mm {'✓' if ok1 else '✗'}(<2mm); "
      f"σ_boot {'✓' if ok2 else '✗'}(<0.2mm); 失败帧处理 "
      f"{'✓' if len(fails) == 0 or res['nf'] >= 2 else '✗'}")
print("总判:" + ("✓ 通过" if ok1 and ok2 else "✗ 未过"))
sys.exit(0 if ok1 and ok2 else 1)
