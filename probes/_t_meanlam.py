# -*- coding: utf-8 -*-
"""_t_meanlam.py:mean 图联合 λ–D 决定性实验(2026-11-08)。

用户点破的不对称:单张可 λ 留空(联合估),mean 模式却强制 λ。
本实验:52 帧 mean 图跑 solve_auto_lambda(同 zoo 单张口径),
判 λ̂ vs 0.7084811、σ_λ、D vs GT 116.860。成→GUI 放开 λ 留空。
"""
import glob
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
from torchfai.phases import PHASE_NAMES

PX1, PX2 = 103.358e-6, 102.530e-6
LAM_GT = 0.7084811
GT = 116.860

D_IR = sorted(glob.glob(r"data\real\alumina\*.edf"))
imgm = np.mean([fabio.open(f).data for f in D_IR], 0).astype(np.float64)
imgm = np.where(np.isfinite(imgm), imgm, 0.0)
print(f"mean {len(D_IR)} 帧")

# solve_auto_lambda(canvas, px, full=True) → dict
r = sc.solve_auto_lambda(imgm, (PX1 + PX2) / 2, full=True)
print("返回 None" if r is None else f"keys: {sorted(r.keys())}")
if r is None:
    sys.exit(1)
lam_hat = r["lam"]
sol = r["sol"]
ph = r["phase"]
sig = r.get("lam_sig")
D_mm = sol[2] * ((PX1 + PX2) / 2) * 1e3
print(f"\nλ̂={lam_hat:.6f}Å (GT {LAM_GT}, Δ{lam_hat-LAM_GT:+.6f}, "
      f"{abs(lam_hat-LAM_GT)/LAM_GT:.3%}) src={r.get('lam_src')}")
if sig is not None:
    print(f"σ_λ={sig:.4f} ({sig/lam_hat:.1%})")
print(f"D={D_mm:.4f}mm (GT {GT}, Δ{D_mm-GT:+.4f})")
print(f"phase={ph} ps={sol[6]} m={sol[3]} rms={sol[4]:.2f}")

ok = (abs(lam_hat - LAM_GT) / LAM_GT < 0.01 and abs(D_mm - GT) < 1.0)
print("\n判决:" + ("✓ mean 联合 λ 可行,GUI 可放开留空" if ok
                       else "✗ 简并域/解不准,维持强制 λ"))
sys.exit(0 if ok else 1)
