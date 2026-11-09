# -*- coding: utf-8 -*-
"""_t_joint8c.py:帧级质量门可行性(2026-11-08)。

joint8 揭示:盲解种子逐帧 Δ 散布 −3.5~+0.8(5/8 帧次优盆),联合
LSQ 无 robust 被拖到 −2.1。验证:每帧 fit_tilt 的 rms5 与 Δ 的
相关性——若相关,可做无 GT 帧级质量门(MAD 剔除)进
joint_solve_frames。
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

z = np.load(ROOT + r"\probes\_t_joint8_seeds.npz", allow_pickle=True)
print(f"{'帧':14} {'D_seed':>8} {'D_ftilt':>9} {'Δ_ftilt':>8} "
      f"{'rms5':>7} {'npts':>5} {'m_seed':>6}")
rows = []
for k in sorted(z.files):
    sd = z[k].item()
    r, s = sd["r"], sd["s"]
    img = fabio.open(os.path.join(AL, k)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    from poni_out import fit_tilt
    do = {}
    tf = fit_tilt(img, r["sol"][0] * s, r["sol"][1] * s, r["sol"][2] * s,
                  QS, LAM, PX1, PX2, sfull=1, nsec=36, mass_rel=0.10,
                  data_out=do)
    if tf is None:
        print(f"{k:14} fit_tilt None")
        continue
    Df = tf["dist"] * 1e3
    rows.append((k, sd["D_mm"], Df, tf["rms5"], len(do["d1"]),
                 r["sol"][3]))
    print(f"{k:14} {sd['D_mm']:8.2f} {Df:9.3f} {Df-GT:+8.3f} "
          f"{tf['rms5']*1e3:7.3f} {len(do['d1']):5} {r['sol'][3]:6}")

arr = np.array([r[3] for r in rows])
Ds = np.array([r[2] for r in rows])
med = np.median(arr)
good = arr <= 1.5 * med
print(f"\nrms5 中位={med*1e3:.3f}mrad;门 1.5× → 保留 {good.sum()}/"
      f"{len(rows)} 帧,其 D 分布 Δ "
      f"[{Ds[good].min()-GT:+.2f},{Ds[good].max()-GT:+.2f}] "
      f"中位 {np.median(Ds[good])-GT:+.2f}")
print(f"  剔除帧 Δ: {[f'{d-GT:+.2f}' for d, g in zip(Ds, good) if not g]}")
