# -*- coding: utf-8 -*-
"""_t1_lamref_test2.py：T1 第二轮——LSQ 收口验证 + mar3450 s=2 λ 盆地映射。

R1 教训：网格+离散评分在简并脊上随机游走（P1M λ̂→1.153）；mar3450 s=3
已复现不出 LaB6（ring_radii 直射束修正改变了 s=3 峰集），s=2+autoC+
λ=0.3705 可出（ps+0.584）。本轮：
  A. P1M 窄窗联合 + LSQ(±20%) 收口 → 期望 λ̂≈1.0326
  B. mar3450 s=2 盆地映射（λ 0.366–0.376 步 0.002, autoC）→ 定盆宽
"""
import os
import sys
import time

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import fabio
import pyFAI.azimuthalIntegrator
from torchfai.phases import phase_qs
from solver_core import solve_auto, solve_auto_lambda, to_canvas


def load_img(path):
    img = np.asarray(fabio.open(path).data, dtype=np.float64)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    return img


def case_a():
    print("===== A. Pilatus1M 窄窗联合 + LSQ 收口 =====", flush=True)
    img = load_img(os.path.join(ZOO, "Pilatus1M.edf"))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
        os.path.join(ZOO, "Pilatus1M.poni"))
    canvas, s, _, _ = to_canvas(img)
    px_m = float(ai.detector.pixel1) * s
    gt_D = ai.dist * 1000.0
    t0 = time.perf_counter()
    r = solve_auto_lambda(
        canvas, px_m, lam_lo=0.90, lam_hi=1.10, n_s=5, auto_center=True,
        refine_stages=((0.04, 7), (0.008, 7)), full=True)
    dt = time.perf_counter() - t0
    ph, sol, lam = r["phase"], r["sol"], r["lam"]
    D = sol[2] * px_m * 1000.0
    print(f"λ_src={r['lam_src']} λ_grid={r['lam_grid']:.4f} "
          f"λ̂={lam:.4f} (期望1.0326, 偏差{abs(lam/1.0326-1)*100:.3f}%) "
          f"σ(λ)/λ={r['lam_sig']}", flush=True)
    print(f"相={ph} m={sol[3]} D@λ̂={D:.2f} ΔD={D-gt_D:+.2f}mm "
          f"[{dt:.0f}s,{r['n_eval']}eval]", flush=True)


def case_b():
    print("\n===== B. mar3450 s=2 λ 盆地映射 =====", flush=True)
    img = load_img(os.path.join(ZOO, "LaB6_260210.mar3450"))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
        os.path.join(ZOO, "Mar3450.poni"))
    canvas, s, _, _ = to_canvas(img, max_side=1730)
    px_m = float(ai.detector.pixel1) * s
    gt_D = ai.dist * 1000.0
    print(f"画布 {canvas.shape} s={s} px={px_m*1000:.2f}mm GT {gt_D:.2f}mm",
          flush=True)
    for lam in (0.3660, 0.3680, 0.3700, 0.3720, 0.3740, 0.3760):
        r = solve_auto(canvas, lam, px_m, cascade=True, auto_center=True)
        if r is None:
            print(f"λ={lam:.4f}: None", flush=True)
            continue
        ph, sol = r
        print(f"λ={lam:.4f}: {ph:8s} m={sol[3]} rms={sol[4]:.2f} "
              f"ps={sol[6]:+.3f} D={sol[2]*px_m*1000:7.1f} "
              f"(Δ{sol[2]*px_m*1000-gt_D:+6.1f}) c=({sol[0]:.0f},{sol[1]:.0f})",
              flush=True)


if __name__ == "__main__":
    case_a()
    case_b()
