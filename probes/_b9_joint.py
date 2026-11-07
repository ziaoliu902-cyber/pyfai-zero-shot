# -*- coding: utf-8 -*-
"""_b9_joint.py:B9 多帧联合标定探针(2026-11-07)。

alumina 51 帧(静态装置,Frelon 半CCD):逐帧盲解 D 序列 → 中位/MAD/
离群;均值帧(像素平均)联合解 vs GT。判决:静态假设(MAD 小)下均值帧
增益;为批量模式「联合标定」按钮提供口径。

GT:al2o3_00_max_51_frames.poni,D=116.86mm,λ=0.7085Å,px=172µm。
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
PX = 172.0
AL = os.path.join(ROOT, "data", "real", "alumina")

frames = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                and f.endswith(".edf"))
print(f"帧数 {len(frames)}  λ={LAM:.4f}Å")

CACHE = os.path.join(ROOT, "probes", "_b9_cache.npz")
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    Ds, ms, phases = z["Ds"], z["ms"], list(z["phases"])
    mean_img = fabio.open(os.path.join(AL, frames[0])).data.astype(np.float64)
    for f in frames[1:]:
        mean_img += fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    mean_img /= len(frames)
    print("(逐帧缓存命中,直接均值帧段)")
else:
    Ds, ms, phases = [], [], []
    mean_img = None
    for i, f in enumerate(frames):
        img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
        mean_img = img if mean_img is None else mean_img + img
        r = sc.solve_image(img, PX * 1e-6, lam=LAM)
        if r is None:
            Ds.append(np.nan); ms.append(0); phases.append("?")
        else:
            Ds.append(r["sol"][2] * r["s"] * PX * 1e-3)
            ms.append(r["sol"][3]); phases.append(r["phase"])
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(frames)}")
    Ds = np.array(Ds)
    np.savez(CACHE, Ds=Ds, ms=np.array(ms), phases=np.array(phases, dtype=object))
    mean_img /= len(frames)
ok = np.isfinite(Ds)

med = float(np.median(Ds[ok]))
mad = float(np.median(np.abs(Ds[ok] - med)))
out_3mad = [frames[i] for i in range(len(frames))
            if ok[i] and abs(Ds[i] - med) > 3 * mad]
out_5mm = [frames[i] for i in range(len(frames))
           if ok[i] and abs(Ds[i] - 116.86) > 5.0]
print(f"\n逐帧: 解出 {ok.sum()}/{len(frames)}  中位 D={med:.2f}mm "
      f"(GT 116.86, Δ{med-116.86:+.2f})  MAD={mad:.2f}mm")
print(f"  离群 >3MAD: {len(out_3mad)} {out_3mad[:8]}")
print(f"  |ΔGT|>5mm:  {len(out_5mm)} {out_5mm[:8]}")
ph_set = sorted(set(phases))
print(f"  相指认: {ph_set}")

r = sc.solve_image(mean_img, PX * 1e-6, lam=LAM)
if r is not None:
    Dm = r["sol"][2] * r["s"] * PX * 1e-3
    print(f"\n均值帧联合: ph={r['phase']} D={Dm:.2f}mm (GT 116.86, "
          f"Δ{Dm-116.86:+.2f}) m={r['sol'][3]} rms={r['sol'][4]:.2f}px")
    gain = "✓ 均值帧更贴 GT" if abs(Dm - 116.86) < abs(med - 116.86) \
        else "✗ 均值帧不优"
    print(f"  vs 逐帧中位: {gain}")
else:
    print("\n均值帧联合: 无解")

print("\n判决口径:MAD≤1mm → 静态装置,均值帧联合标定增益可用;")
print("         MAD 大或离群多 → 装置在动,均值帧会糊,只报漂移曲线。")
