# -*- coding: utf-8 -*-
"""_t_joint8g.py:生产种子口径终验(2026-11-08)。

joint8f:52 帧 mean 三臂全 +1.4~1.9 → 116.79 是 phase_fit 的
**逐帧解均值**(锁相+px平均+旧三关 auto_center/refine_2d=False;
弱衍射帧上默认全开的 2D 精化反把解拉进次优盆)。
终验:8 帧逐帧锁相+px平均+旧三关 → D 均值≈116.8、MAD≈0.4
→ 该口径做 BatchWorker 联合种子(快:无 2D 精化)。
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
PX = (PX1 + PX2) / 2
LAM = 0.7084811
GT = 116.860
QS = phase_qs("Al2O3")
AL = os.path.join(ROOT, "data", "real", "alumina")
frames_all = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                    and f.endswith(".edf"))
sample = [frames_all[i] for i in (0, 6, 12, 18, 24, 30, 36, 42)]
OLD = dict(cascade=True, auto_center=False, refine_2d=False)

cpts, Ds = [], []
print("8 帧逐帧(锁相+px平均+旧三关)种子 → fit_tilt → 联合:")
for k, f in enumerate(sample):
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    r = sc.solve_image(img, PX, lam=LAM, phases=["Al2O3"], solve_kw=OLD)
    if r is None:
        print(f"  [{k+1}/8] {f}: 无解")
        cpts.append(None)
        continue
    s = r["s"]
    D = r["sol"][2] * s * PX * 1e3
    Ds.append(D)
    c = sc.cpts_of_frame(img, r["sol"][0] * s, r["sol"][1] * s,
                         r["sol"][2] * s, PX1, PX2, LAM, QS)
    cpts.append(c)
    print(f"  [{k+1}/8] {f}: D={D:.2f} (Δ{D-GT:+.2f}) "
          f"→ {len(c['d1']) if c else 0}pt", flush=True)

Ds = np.array(Ds)
print(f"\n逐帧:n={len(Ds)} 均值={Ds.mean():.3f} (Δ{Ds.mean()-GT:+.3f}) "
      f"MAD={np.median(np.abs(Ds-np.median(Ds))):.3f}")
res = sc.joint_solve_frames(cpts, PX1, PX2, boot=6)
assert res is not None
Dj = res["dist"] * 1e3
print(f"联合:D={Dj:.3f}mm (Δ{Dj-GT:+.3f}) "
      f"rms={res['rms']*1e3:.2f}mrad σ_boot={res['sigma_boot']:.3f}mm")
ok = abs(Dj - GT) < 1.0
print("总判:" + ("✓ 旧三关种子链达标(Δ<1mm)" if ok else "✗"))
sys.exit(0 if ok else 1)
