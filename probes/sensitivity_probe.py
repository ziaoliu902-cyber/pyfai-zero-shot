# -*- coding: utf-8 -*-
"""sensitivity_probe.py：W7 误差-参数敏感性分析（doc 12 数据源）。

Part A 先验误差传播系数（ceria 真实域，理论斜率：λ→−1, Q→−1, PX→+1，
小角近似下三者皆 1:1 传播）：
  d(lnD)/d(lnλ), d(lnD)/d(lnQ), d(lnD)/d(lnPX)  @ ±0.2/0.5/1/2%
Part B 噪声敏感性：y = Poisson(img·α)/α（均值不变，方差/α），
α ∈ {0.2, 0.5, 1, 2, 5} → 生产路径 ΔD——曝光下限在哪里。

用法: python probes/sensitivity_probe.py
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI

from solver_core import ring_radii, solve, to_canvas
from torchfai.phases import phase_qs

LAM0 = 0.972386
Q0 = phase_qs("CeO2")[:8]
FILES = [f"ceria_{d}_1_0001" for d in (150, 200, 250, 300, 350, 400, 450)]


def fit_D(img, lam, Q, px_scale=1.0):
    """生产经典路径：to_canvas + 几何中心 solve → D(mm)。"""
    canvas, s, cx0, cy0 = to_canvas(img)
    px_m = 172e-6 * s * px_scale
    peaks, prof, bin_px, good = ring_radii(canvas, cx0, cy0)
    sol = solve(peaks, cx0, cy0, 1.0 / px_m, Q, lam, px_m,
                d_range=(0.05, 3.0), canvas=canvas,
                prof_ctx=(prof, bin_px, good), rms_floor=1.8)
    if sol is None:
        return None
    return sol[2] * px_m * 1000


imgs = {}
gts = {}
for f in FILES:
    raw = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/{f}.cbf")).data.astype(np.float64)
    # CBF 可能带 NaN/负哨兵 → 泊松重采样前清洗（生产 ring_radii 对此
    # 不敏感，但 numpy.poisson 要求 lam≥0）
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    imgs[f] = np.clip(raw, 0, None)
    gts[f] = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/{f}.poni")).dist * 1000

# ---------- Part A：先验误差传播系数 ----------
def perturbed(tag, eps):
    """只动一个旋钮：lam / Q / px_scale。返回 fit_D kwargs。"""
    kw = dict(lam=LAM0, Q=Q0, px_scale=1.0)
    if tag == "lam":
        kw["lam"] = LAM0 * (1 + eps)
    elif tag == "q":
        kw["Q"] = Q0 * (1 + eps)
    else:
        kw["px_scale"] = 1 + eps
    return kw


print("== Part A：先验误差传播（ceria 七档，理论斜率 λ/Q→−1, PX→+1）==")
print(f"{'ε':>6} | {'斜率(λ)':>10} {'斜率(Q)':>10} {'斜率(PX)':>10}")
for eps in (0.002, 0.005, 0.01, 0.02):
    sl = {k: [] for k in ("lam", "q", "px")}
    for f in FILES:
        D0 = fit_D(imgs[f], LAM0, Q0)
        if D0 is None:
            continue
        for tag in ("lam", "q", "px"):
            Dp = fit_D(imgs[f], **perturbed(tag, eps))
            Dm = fit_D(imgs[f], **perturbed(tag, -eps))
            if Dp is None or Dm is None:
                continue
            s_hi = np.log(Dp / D0) / np.log(1 + eps)
            s_lo = np.log(Dm / D0) / np.log(1 - eps)
            sl[tag].append((s_hi + s_lo) / 2)
    print(f"{eps*100:5.1f}% | "
          f"{np.mean(sl['lam']):+10.3f} {np.mean(sl['q']):+10.3f} "
          f"{np.mean(sl['px']):+10.3f}")

# ---------- Part B：噪声敏感性 ----------
print("\n== Part B：噪声敏感性（Poisson 重采样 y=P(img·α)/α，均值不变）==")
print(f"{'α':>6} | {'成功':>6} {'中位Δmm':>9} {'最差Δmm':>9} {'中位环数':>8}")
rng = np.random.default_rng(3)
for a in (5.0, 2.0, 1.0, 0.5, 0.2, 0.05):
    res = []
    for f in FILES:
        y = rng.poisson(imgs[f] * a) / a
        canvas, s, cx0, cy0 = to_canvas(y)
        px_m = 172e-6 * s
        peaks, prof, bin_px, good = ring_radii(canvas, cx0, cy0)
        sol = solve(peaks, cx0, cy0, 1.0 / px_m, Q0, LAM0, px_m,
                    d_range=(0.05, 3.0), canvas=canvas,
                    prof_ctx=(prof, bin_px, good), rms_floor=1.8)
        res.append(None if (sol is None or sol[3] < 3)
                   else (sol[2] * px_m * 1000 - gts[f], sol[3]))
    ok = [r for r in res if r is not None]
    mm = [abs(r[0]) for r in ok]
    print(f"{a:5.1f}× | {len(ok)}/7   "
          f"{np.median(mm):9.2f} {max(mm) if mm else float('nan'):9.2f} "
          f"{np.median([r[1] for r in ok]) if ok else float('nan'):8.1f}")
