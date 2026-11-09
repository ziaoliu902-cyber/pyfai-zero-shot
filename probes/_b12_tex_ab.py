# -*- coding: utf-8 -*-
"""_b12_tex_ab.py:B12① 织构域 A/B(2026-11-07)——splitG 启用判决。

B11d:ceria/P2M(峰对称域)splitG 中性 → 默认关。
本探针:alumina 51 帧(织构)抽 9 帧,锁相 Al2O3(λ=12.398/17.5,
px=103.358µm,GT D=116.86mm),splitG on/off 两臂全管线 solve_image。
判据(文档既定):splitG 臂 |ΔGT| 均值或 MAD 改善 >0.05mm → 开 _SPLIT_G。

用法:python _b12_tex_ab.py [start:end]   # 帧切片,默认 0:9
每帧结果即写 _b12_tex_rows.npy(list of (fname, dict))可断点续跑;
无参全 9 帧跑完打印判决。
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

LAM = 12.398 / 17.5
PX_UM = 103.358
GT = 116.860
AL = os.path.join(ROOT, "data", "real", "alumina")
ROWS = os.path.join(ROOT, "probes", "_b12_tex_rows.npy")

frames = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                and f.endswith(".edf"))
sample = [frames[i] for i in (0, 6, 12, 18, 24, 30, 36, 42, 48)]
a, b = (int(x) for x in sys.argv[1].split(":")) if len(sys.argv) > 1 else (0, 9)
ROWS = os.path.join(ROOT, "probes", f"_b12_tex_rows_{a}_{b}.npy")
print(f"B12① 织构 A/B 帧[{a}:{b}] 锁相 Al2O3 λ={LAM:.6f} "
      f"px={PX_UM}µm GT={GT}mm", flush=True)

rows = list(np.load(ROWS, allow_pickle=True)) if os.path.exists(ROWS) else []
done = {r[0] for r in rows}
for f in sample[a:b]:
    if f in done:
        print(f"  {f}: 已有,跳过", flush=True)
        continue
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    out = {}
    for lbl, flag in (("splitG", True), ("parab", False)):
        sc._SPLIT_G = flag
        sc.clear_masks()
        r = sc.solve_image(img, PX_UM * 1e-6, lam=LAM, phases=["Al2O3"])
        out[lbl] = None if r is None else dict(
            D=r["sol"][2] * r["s"] * PX_UM * 1e-3,
            m=r["m"], rms=r["rms"])
    sc._SPLIT_G = False
    d1, d2 = out["parab"], out["splitG"]
    line = f"{f}: "
    for lbl in ("parab", "splitG"):
        d = out[lbl]
        line += (f"{lbl} Δ={d['D']-GT:+6.3f} rms={d['rms']:4.2f} m={d['m']:2}  "
                 if d else f"{lbl} 无解  ")
    print(line, flush=True)
    rows.append((f, out))
    np.save(ROWS, np.array(rows, dtype=object), allow_pickle=True)

if b - a == 9 or (len(rows) >= 9 and b >= 9):
    print("\n== 判决(全部 9 帧;分片模式只存不判,汇总用 _b12_tex_sum.py) ==")
    for lbl in ("parab", "splitG"):
        ds = np.array([o[lbl]["D"] - GT for _, o in rows if o[lbl]])
        if len(ds):
            print(f"  {lbl:7} n={len(ds)} 均值Δ={ds.mean():+.3f}mm "
                  f"|Δ|均值={np.abs(ds).mean():.3f}mm "
                  f"MAD={np.median(np.abs(ds-np.median(ds))):.3f}mm")
    dp = np.abs(np.array([o["parab"]["D"] - GT for _, o in rows
                          if o["parab"]])).mean()
    ds_ = np.abs(np.array([o["splitG"]["D"] - GT for _, o in rows
                           if o["splitG"]])).mean()
    gain = dp - ds_
    print(f"\n  |Δ|均值增益:{gain:+.3f}mm")
    print("  判定:" + ("增益>0.05mm → 开启 _SPLIT_G=True" if gain > 0.05
          else ("劣化 → 保持关" if gain < -0.05 else "中性 → 保持关")))
