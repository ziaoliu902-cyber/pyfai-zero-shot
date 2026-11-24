# -*- coding: utf-8 -*-
"""几何前向模型与五参数最小二乘（自写，不依赖 pyFAI 的拟合器）。

坐标约定与 pyFAI 一致（用于和 pyFAI 交叉验证）：
    探测器平面点 (p1, p2, p3) = (Y, X, D)
    R = R3(rot3)·R2(rot2)·R1(rot1)
    2θ = atan2( hypot(t1, t2), t3 )
其中 rot1/rot2 矩阵为左旋（与 pyFAI 相同）。
"""

import numpy as np


def rotation_matrix(rot1, rot2, rot3=0.0):
    c1, s1 = np.cos(rot1), np.sin(rot1)
    c2, s2 = np.cos(rot2), np.sin(rot2)
    c3, s3 = np.cos(rot3), np.sin(rot3)
    r1 = np.array([[1.0, 0.0, 0.0], [0.0, c1, s1], [0.0, -s1, c1]])
    r2 = np.array([[c2, 0.0, -s2], [0.0, 1.0, 0.0], [s2, 0.0, c2]])
    r3 = np.array([[c3, -s3, 0.0], [s3, c3, 0.0], [0.0, 0.0, 1.0]])
    return r3 @ r2 @ r1


def tth_from_pixel(x, y, dist, poni1, poni2, rot1, rot2, px, px2=None):
    """像素坐标 (x=列, y=行) → 2θ（弧度）。支持标量或数组及非等方像素 px2。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    px1 = float(px)
    px2_ = px1 if px2 is None else float(px2)
    p1 = (y + 0.5) * px1 - poni1
    p2 = (x + 0.5) * px2_ - poni2
    p3 = np.full_like(p1, dist)
    rot = rotation_matrix(rot1, rot2)
    t1 = rot[0, 0] * p1 + rot[0, 1] * p2 + rot[0, 2] * p3
    t2 = rot[1, 0] * p1 + rot[1, 1] * p2 + rot[1, 2] * p3
    t3 = rot[2, 0] * p1 + rot[2, 1] * p2 + rot[2, 2] * p3
    return np.arctan2(np.hypot(t1, t2), t3)


def fit_geometry(xy, tth_obs, p0, px, weights=None, bounds=None, px2=None):
    """最小二乘拟合 (dist, poni1, poni2, rot1, rot2)。

    xy: (N, 2) 像素坐标（x=列, y=行）
    tth_obs: (N,) 每个像素所属环的理论 2θ（弧度）
    p0: 初值 [dist, poni1, poni2, rot1, rot2]（米/米/米/弧度/弧度）
    """
    from scipy.optimize import least_squares

    x = np.asarray(xy[:, 0], float)
    y = np.asarray(xy[:, 1], float)
    w = np.ones_like(tth_obs) if weights is None else np.asarray(weights, float)

    def resid(p):
        d, q1, q2, r1, r2 = p
        return (tth_from_pixel(x, y, d, q1, q2, r1, r2, px, px2=px2) - tth_obs) * w

    if bounds is None:
        # 物理约束：距离正、倾角小于 5°
        bounds = ([p0[0] * 0.5, p0[1] - 0.05, p0[2] - 0.05, -0.09, -0.09],
                  [p0[0] * 1.5, p0[1] + 0.05, p0[2] + 0.05, 0.09, 0.09])
    res = least_squares(resid, np.asarray(p0, float), bounds=bounds,
                        method="trf", loss="soft_l1", f_scale=1e-4)
    # 协方差（近似）
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (2 * res.cost / max(len(res.fun) - 5, 1))
        sig = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sig = np.full(5, np.nan)
    return dict(p=res.x, sigma=sig, rms=float(np.sqrt(np.mean(res.fun ** 2))),
                cond=float(np.linalg.cond(J)), success=bool(res.success))
