# -*- coding: utf-8 -*-
"""_t_spl_joint.py:④b 联合层低阶畸变基实验(2026-11-08,实验性非默认)。

问题:向导口径(方形 px+无 spline)52 帧 joint D=118.00 vs GT 116.86
(Δ+1.14mm)。batch 全口径(分立 px)116.73(Δ−0.13)。差 ~1.3mm 的
来源:方形 px 近似+未知畸变。

实验:joint_solve_frames 的前向 tth(D,rot,p1,p2) 外加低阶径向畸变
δr = k3·r³ + k5·r⁵(r=归一化半径),全局拟合 k3,k5——看
(a) D 是否收敛到 GT±0.3(畸变被吸收,系统差修复)
(b) 还是 k 与 D 简并乱跑(负结果,留档)
判定:D 回到 116.86±0.3 → 值得进生产;k|大|而 D 不动/更偏 → 简并实锤。
"""
import glob
import os
import sys

import numpy as np
import fabio
from scipy.optimize import least_squares

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probes"))
os.chdir(ROOT)

import solver_core as sc
from torchfai.phases import phase_qs
from poni_out import make_pyfai_fwd, fit_tilt

PX1, PX2 = 103.358e-6, 102.530e-6
LAM = 0.7084811
GT = 116.860

files = sorted(glob.glob(r"data\real\alumina\*.edf"))[:52]
imgs = []
for f in files:
    a = fabio.open(f).data
    a = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    imgs.append(a)
print(f"{len(imgs)} 帧载入")

# ── 与向导同口径的方形 px 种子(复现 D=118 起点) ──
pxs = (PX1 + PX2) / 2
seed = sc.b_route_mean_seed(iter(imgs), pxs, pxs, LAM, "Al2O3")
assert seed is not None
cx_n = seed["poni2"] / pxs - 0.5
cy_n = seed["poni1"] / pxs - 0.5
D_n = seed["dist"] / pxs
print(f"seed D={seed['dist']*1e3:.2f}mm")

qs = phase_qs("Al2O3")
cpts = []
for im in imgs:
    c = sc.cpts_of_frame(im, cx_n, cy_n, D_n, pxs, pxs, LAM, qs)
    if c is not None:
        cpts.append(c)
print(f"cpts {len(cpts)} 帧有效")

D1 = np.concatenate([c["d1"] for c in cpts])
D2 = np.concatenate([c["d2"] for c in cpts])
TT = np.concatenate([c["tth"] for c in cpts])
FR = np.concatenate([[k] * len(c["d1"]) for k, c in enumerate(cpts)])
W = np.concatenate([c["w"] for c in cpts]); W = W / W.mean()
NX, NY = imgs[0].shape[1], imgs[0].shape[0]
RMAX = np.hypot(NX * pxs, NY * pxs) / 2

fwd = make_pyfai_fwd(pxs, pxs)

def resid(p, with_k):
    k3, k5 = (p[-2], p[-1]) if with_k else (0.0, 0.0)
    x = (D1 + 0.5 - NX / 2) * pxs
    y = (D2 + 0.5 - NY / 2) * pxs
    r = np.hypot(x, y) / RMAX
    dr = (k3 * r ** 3 + k5 * r ** 5) * RMAX      # 径向偏移(米)
    scale = 1.0 + dr / np.maximum(r * RMAX, 1e-9)
    d1k = ((D1 + 0.5 - NX / 2) * scale + NX / 2 - 0.5)
    d2k = ((D2 + 0.5 - NY / 2) * scale + NY / 2 - 0.5)
    base = fwd(p[0], p[3 + 2 * FR], p[4 + 2 * FR], p[1], p[2], d1k, d2k)
    return (base - TT) * W

# ── 基线(无畸变,k 固定 0)──
p0 = np.concatenate([[seed["dist"], 0.0, 0.0],
                     np.array([[c["p1"], c["p2"]] for c in cpts]).ravel()])
f0 = least_squares(resid, p0, args=(False,), method="lm", max_nfev=4000)
D0 = f0.x[0] * 1e3
print(f"\n基线(无畸变基): D={D0:.3f}mm (Δ{D0-GT:+.3f})")

# ── +k3/k5 联合 ──
p0k = np.concatenate([p0, [0.0, 0.0]])
fk = least_squares(resid, p0k, args=(True,), method="lm", max_nfev=8000)
Dk = fk.x[0] * 1e3
print(f"畸变基联合: D={Dk:.3f}mm (Δ{Dk-GT:+.3f}) k3={fk.x[-2]:+.2e} "
      f"k5={fk.x[-1]:+.2e}")

ok = abs(Dk - GT) < 0.3
print("\n判决:" + ("✓ 畸变被吸收,D 修复——值得进生产(需防简并正则)"
      if ok else "✗ k–D 简并未破/未修复——负结果留档,维持显式 spline 口径"))
sys.exit(0 if ok else 1)
