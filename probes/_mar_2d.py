# -*- coding: utf-8 -*-
"""mar3450：2D 逐环圆拟合原型（不做径向平均）。

对每条预测环：
  1. 在当前中心下取 |r - r_i| < 3px 的像元，背景阈值选亮像元；
  2. Kasa 代数圆拟合（2.5σ 剪枝两遍）→ 该环的 (cx_i, cy_i, r_i)；
  3. 环间按环质量加权聚合中心；
  4. 用各环半径对 D 做加权最小二乘：r_i = D · tan(2θ(q_i,λ))/px。
迭代 3 轮。环半径初值来自 1D 解（LaB6, D0≈220.8mm），只用来选带。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")

import fabio
import pyFAI.azimuthalIntegrator
from torchfai.phases import phase_qs

sys.path.insert(0, ROOT + r"\probes")
import solver_core as SC


def load_native():
    img = np.asarray(fabio.open(os.path.join(ZOO, "LaB6_260210.mar3450")).data,
                     float)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    return img


def kasa(kx, ky):
    A = np.stack([kx, ky, np.ones_like(kx)], 1)
    b = kx * kx + ky * ky
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = sol
    r2 = c + (a * a + bb * bb) / 4.0
    if r2 <= 0:
        return None
    return a / 2.0, bb / 2.0, float(np.sqrt(r2))


def fit_rings(img, c0, radii0, half=3.0, trim=2.5):
    H, W = img.shape
    yy, xx = np.indices(img.shape)
    cx, cy = c0
    out = []
    for qi, r0 in radii0:
        if r0 < 30:
            continue
        r_all = np.hypot(yy - cy, xx - cx)
        band = np.abs(r_all - r0) < half
        iy, ix = np.nonzero(band)
        if iy.size < 200:
            continue
        v = img[iy, ix]
        base = float(np.percentile(v, 35))
        peak = float(np.percentile(v, 98))
        if peak <= base:
            continue
        keep = v > base + 0.35 * (peak - base)
        px, py = ix[keep].astype(float), iy[keep].astype(float)
        if px.size < 150:
            continue
        res = kasa(px, py)
        if res is None:
            continue
        for _ in range(2):
            rr = np.abs(np.hypot(px - res[0], py - res[1]) - res[2])
            sg = rr.std()
            m = rr < trim * max(sg, 0.5)
            if m.sum() < 150 or m.all():
                break
            px, py = px[m], py[m]
            r2 = kasa(px, py)
            if r2 is None:
                break
            res = r2
        rr = np.abs(np.hypot(px - res[0], py - res[1]) - res[2])
        out.append((res[0], res[1], res[2], px.size, float(rr.std()), qi))
    return out


def solve_d(radii, q, lam, px_m):
    tt = 2 * np.arcsin(np.clip((lam * 1e10) * q / (4 * np.pi), 0, 0.9999))
    k = np.tan(tt) / px_m
    w = 1.0 / np.maximum(radii, 1.0) ** 2   # 大半径权重高（相对精度）
    D = float(np.sum(w * radii * k) / np.sum(w * k * k)) * 1000.0
    return D, k


def main():
    img = load_native()
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
        os.path.join(ZOO, "Mar3450.poni"))
    f2d = ai.getFit2D()
    gt = (f2d["centerX"], f2d["centerY"])
    q = np.asarray(phase_qs("LaB6"), float)
    lam = ai.wavelength
    px_m = ai.detector.pixel1

    # 初值：1D 解给 D0=220.8mm；环半径按 D0 预测
    D0 = 0.2208
    tt = 2 * np.arcsin(np.clip((lam * 1e10) * q / (4 * np.pi), 0, 0.9999))
    radii0 = D0 * np.tan(tt) / px_m
    radii0 = [(i, r) for i, r in enumerate(radii0) if 150 < r < 1700]

    for tag, c0 in (("GT", gt), ("s3-refine", (1719.45, 1708.68)), ("s3+5", (1724.45, 1713.68))):
        if c0 is None:
            v = SC.gradient_vote_center(img[::4, ::4])
            c0 = (v[0] * 4, v[1] * 4)
        c = c0
        for it in range(3):
            fits = fit_rings(img, c, radii0)
            if len(fits) < 3:
                print(tag, "iter", it, "fits", len(fits), "-> stop")
                break
            w = np.array([f[3] for f in fits], float)
            nx = float(np.sum([f[0] * wi for f, wi in zip(fits, w)]) / w.sum())
            ny = float(np.sum([f[1] * wi for f, wi in zip(fits, w)]) / w.sum())
            radii0 = [(f[5], f[2]) for f in fits]
            c = (0.5 * nx + 0.5 * c[0], 0.5 * ny + 0.5 * c[1])
            print(f"{tag} it{it}: c=({c[0]:.2f},{c[1]:.2f}) dGT="
                  f"{np.hypot(c[0]-gt[0], c[1]-gt[1]):.2f} rings={len(fits)}")
        if len(fits) >= 3:
            D, k = solve_d(np.array([f[2] for f in fits]),
                           q[:len(fits)], lam, px_m)
            print(f"{tag}: D={D:.2f}mm (GT {ai.dist*1000:.2f}, "
                  f"dD={D-ai.dist*1000:+.2f}) center=({c[0]:.2f},{c[1]:.2f}) "
                  f"dGT={np.hypot(c[0]-gt[0], c[1]-gt[1]):.2f}px")


if __name__ == "__main__":
    main()



