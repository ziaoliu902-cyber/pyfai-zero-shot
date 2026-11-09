# -*- coding: utf-8 -*-
"""_b11_budget.py:B11 误差预算探针(2026-11-07,科学精度分解)。

问题:盲解 ΔD 已到 0.03~0.1mm 级,科学上还能压吗?被什么卡?
方法:GT 几何(ceria_200 poni)合成完美环图 → 逐项注入误差源 →
全管线盲解(solve_image,与生产同口径)→ ΔD 响应 = 灵敏度。

注入项(每次一项,其余完美):
  A 数值地基(无注入)          → 算法自身偏差
  B λ 输入偏 -1e-3 相对        → λ-D 乘积简并(同步辐射能量标定精度)
  C 像素输入偏 -1e-4 相对      → 探测器像素标定
  D 峰形偏斜 +0.3px(切趾)     → 织构/倾角的选择效应(径向不对称)
  E 面内倾角 rot1=1°(画图侧)  → 几何模型差(求解器 fit_tilt 层在盲解路径)
  F 径向畸变 1%(r→r(1+0.01(r/rmax)²)) → spline 未校正
  G 方位遮 40%(部分环!)       → 1D 谱方位平均的选择效应
  H 统计极限(计数×1/10)       → 散粒噪声
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
from poni_out import make_pyfai_fwd
from torchfai.phases import phase_qs

rng = np.random.default_rng(7)

PX_UM = 172.0
PX = PX_UM * 1e-6
LAM = 0.972386
D_MM = 200.19
H, W = 2527, 2463
Q = phase_qs("CeO2")[:9]


def synth(lam=LAM, px=PX, skew_px=0.0, rot1=0.0, distort=0.0,
          az_mask=1.0, counts=3000.0):
    """GT 几何 + CeO2 9 环;方位高斯环 + 泊松。az_mask=可见方位占比。"""
    d1, d2 = np.meshgrid(np.arange(W) * px, np.arange(H) * px)  # m
    # 中心与 D 用 ceria_200 近似 GT(中心偏画布心一点)
    cx, cy = W * px * 0.495, H * px * 0.502
    r = np.hypot(d1 - cx, d2 - cy)
    rmax = r.max()
    if distort:
        r = r * (1.0 + distort * (r / rmax) ** 2)
        d1 = cx + (d1 - cx) * (1.0 + distort * (r / rmax) ** 2)
        d2 = cy + (d2 - cy) * (1.0 + distort * (r / rmax) ** 2)
    D_m = D_MM * 1e-3
    tt_q = 2 * np.arcsin(np.clip(Q * lam / (4 * np.pi), 0, 0.999))
    # 面内倾角一阶:2θ → 2θ + rot1·cos(方位)(小角近似,1° 量级足够)
    phi = np.arctan2(d2 - cy, d1 - cx)
    img = np.zeros((H, W))
    sig = 1.5 * px
    for k, tq in enumerate(tt_q):
        # 环半径(D·tan2θ 画布换算到米)
        rr = D_m * np.tan(tq)
        dr = r - rr
        # 峰形偏斜:半幅不对称(左窄右宽)
        if skew_px:
            w = sig * np.where(dr < 0, 1 - skew_px / 1.5, 1 + skew_px / 1.5)
        else:
            w = sig
        img += counts * np.exp(-0.5 * (dr / w) ** 2)
    if az_mask < 1.0:
        keep = (np.cos(phi - np.pi * az_mask) > np.cos(np.pi * (1 - az_mask)))
        img = np.where(keep, img, 0.0)
    img = rng.poisson(img).astype(float)
    return img


def solve(tag, img, lam=LAM, px_um=PX_UM):
    r = sc.solve_image(img, px_um * 1e-6, lam=lam)
    if r is None:
        print(f"{tag:28} 无解")
        return None
    D = r["sol"][2] * r["s"] * px_um * 1e-3
    d = D - D_MM
    print(f"{tag:28} D={D:8.2f}mm  Δ={d:+7.3f}mm  m={r['sol'][3]} "
          f"rms={r['sol'][4]:.2f}px")
    return d


def _main():
    print("=" * 76)
    print("B11 误差预算:合成完美环图 → 注入 → 盲解 ΔD 响应(GT D=200.19mm)")
    print("=" * 76)
    rows = []
    rows.append(("A 数值地基(无注入)", solve("A 数值地基", synth())))
    rows.append(("B λ偏-1e-3", solve("B λ-1e-3",
                 synth(), lam=LAM * (1 - 1e-3))))
    rows.append(("C 像素偏-1e-4", solve("C 像素-1e-4",
                 synth(), px_um=PX_UM * (1 - 1e-4))))
    rows.append(("D 峰形偏斜+0.3px", solve("D 偏斜", synth(skew_px=0.3))))
    rows.append(("E 倾角rot1=1°", solve("E 倾角1°", synth(rot1=1.0))))
    rows.append(("F 径向畸变1%", solve("F 畸变1%", synth(distort=0.01))))
    rows.append(("G 方位遮40%(部分环)", solve("G 部分环", synth(az_mask=0.6))))
    rows.append(("H 计数×1/10", solve("H 低计数", synth(counts=300.0))))

    print("\n== 灵敏度表(每项→ΔD;对比实测总误差 0.03~0.1mm) ==")
    for tag, d in rows:
        if d is None:
            continue
        print(f"  {tag:26} ΔD={d:+8.3f}mm")


if __name__ == "__main__":
    _main()
