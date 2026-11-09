# -*- coding: utf-8 -*-
"""_b11b_skew.py:B11b 峰形偏斜对抗判决(2026-11-07)。

B11 结论:偏斜 0.3px → 1D 谱 D 偏 +0.124mm(织构族系统误差主源)。
假设:偏斜是方位选择效应(r(φ) 椭圆 × I(φ) 加权平均 ≠ R),
     控制点层(扇区峰位→谐波拟合 R_j)天然无偏 → 用控制点 R_j
     定 D 应消掉该项。

判决:同一注入图,两个口径的 ΔD:
  (a) solve_image 全管线(1D 峰位驱动)   —— 已知 +0.124mm
  (b) GT 中心/D 种子 → fit_tilt 控制点 → 谐波拟合 R_j·(4π/λ…)
      → 最小二乘 D_cp(r_j = D·tan2θ_j)
注入:偏斜 0.3 / 0.6px 两档 + 无偏斜对照。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc
from poni_out import fit_tilt, make_pyfai_fwd
from torchfai.phases import phase_qs
from _b11_budget import synth, PX_UM, LAM, D_MM, Q

PX = PX_UM * 1e-6


def cp_D(img):
    """GT 种子 → fit_tilt 控制点 → R_j 谐波中位 → D 最小二乘。"""
    H, W = img.shape
    cx, cy = W * 0.495, H * 0.502
    D_px = D_MM * 1e-3 / PX
    do = {}
    tf = fit_tilt(img, cx, cy, D_px, Q, LAM, PX, PX,
                  sfull=1, nsec=36, mass_rel=0.10, bump=1.5, data_out=do)
    if tf is None or len(do.get("d1", [])) < 30:
        return None, None
    # 控制点 → 每环半径(谐波 1 阶拟合的常数项即无偏 R_j)
    # 注意:do 的 d1/d2 是全分辨率像素索引;poni1_0/poni2_0 是米 → 换像素
    d1, d2, qidx = do["d1"], do["d2"], do["qidx"]
    phi = np.arctan2(d2 - do["poni2_0"] / PX, d1 - do["poni1_0"] / PX)
    rr = np.hypot(d1 - do["poni1_0"] / PX, d2 - do["poni2_0"] / PX)
    tt_q = 2 * np.arcsin(np.clip(Q * LAM / (4 * np.pi), 0, 0.999))
    Rs, ws = [], []
    for j in np.unique(qidx):
        m = qidx == j
        if m.sum() < 8:
            continue
        A = np.column_stack([np.ones(m.sum()), np.cos(2 * phi[m]),
                             np.sin(2 * phi[m])])
        c, *_ = np.linalg.lstsq(A, rr[m], rcond=None)
        Rs.append(c[0]); ws.append(m.sum())
    Rs = np.array(Rs); ws = np.array(ws, float)
    if len(Rs) < 4:
        return None, None, None
    tj = tt_q[:len(Rs)]
    # r_j = D_px·tan(2θ_j) 加权最小二乘
    D_cp = float((ws * Rs * np.tan(tj)).sum() / (ws * np.tan(tj) ** 2).sum())
    return D_cp * PX * 1e3, len(Rs), tf["dist"] * 1e3


print("=" * 70)
print("B11b 偏斜对抗:1D 谱 D vs 控制点 D(同图三口径)")
print("=" * 70)
print(f"{'注入':12}{'1D谱ΔD':>12}{'谐波R_jΔD':>12}{'tf5参ΔD':>11}{'环':>4}")
for tag, sk in (("无偏斜", 0.0), ("偏斜0.3px", 0.3), ("偏斜0.6px", 0.6)):
    img = synth(skew_px=sk)
    r = sc.solve_image(img, PX, lam=LAM)
    d1_ = (r["sol"][2] * r["s"] * PX * 1e3 - D_MM) if r else float("nan")
    Dcp, nj, Dtf = cp_D(img)
    d2_ = (Dcp - D_MM) if Dcp else float("nan")
    d3_ = (Dtf - D_MM) if Dtf else float("nan")
    print(f"{tag:12}{d1_:>+10.3f}mm{d2_:>+10.3f}mm{d3_:>+9.3f}mm"
          f"{nj if nj else 0:>4}")
