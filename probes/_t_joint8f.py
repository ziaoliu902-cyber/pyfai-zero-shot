# -*- coding: utf-8 -*-
"""_t_joint8f.py:mean-frame 口径三因子定析(2026-11-08)。

phase_fit alumina(116.79 出处)口径:px=(px1+px2)/2、锁相、
solve_kw 旧三关(cascade=True, auto_center=False, refine_2d=False)。
joint8e(52 帧 mean,+2.24)口径:px=px1、默认 solve_kw(全开)。
三臂定析:复刻 / 换 px / 换 solve_kw。
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

PX1, PX2 = 103.358e-6, 102.530e-6
LAM = 0.7084811
GT = 116.860
AL = os.path.join(ROOT, "data", "real", "alumina")
frames = sorted(f for f in os.listdir(AL) if f.startswith("al2o3_")
                and f.endswith(".edf"))
acc = None
for f in frames:
    img = fabio.open(os.path.join(AL, f)).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    acc = img.copy() if acc is None else acc + img
mean_img = acc / len(frames)
print(f"52 帧 mean-frame 三臂(λ={LAM:.7f} 锁相 Al2O3):")

OLD = dict(cascade=True, auto_center=False, refine_2d=False)
for tag, px, kw in (("a 复刻(px平均+旧三关)", (PX1 + PX2) / 2, OLD),
                    ("b px平均+默认全开", (PX1 + PX2) / 2, None),
                    ("c px1+旧三关", PX1, OLD)):
    r = sc.solve_image(mean_img, px, lam=LAM, phases=["Al2O3"],
                       solve_kw=kw)
    if r is None:
        print(f"  [{tag}] 无解")
        continue
    s = r["s"]
    D = r["sol"][2] * s * px * 1e3
    print(f"  [{tag}] D={D:.3f}mm (Δ{D-GT:+.3f}) m={r['sol'][3]} "
          f"rms={r['sol'][4]:.3f}px", flush=True)
