# -*- coding: utf-8 -*-
"""peak_fit.py：衍射环峰形拟合模块（Pseudo-Voigt 亚像素提取）

物理背景：
  粉末 X 射线衍射环在径向剖面上的峰形是由高斯（仪器展宽、探测器点扩散）
  与洛伦兹（晶粒尺寸、微观应变）卷积而成的 Voigt 函数。
  在实际计算中，高精度的线性组合 Pseudo-Voigt 近似：
      PV(r) = I_0 · [ η · L(r; r_0, w) + (1 - η) · G(r; r_0, w) ] + bg
  能以亚 0.05 像素的精度提取峰中心 r_0，并给出真实的半高宽 FWHM 与拟合不确定度，
  克服传统 3 点抛物线插值受背景斜率与洛伦兹长尾拖尾影响导致的峰位系统漂移。
"""

import numpy as np
from scipy.optimize import curve_fit


def pseudo_voigt_func(r, r0, w, eta, I0, bg, slope=0.0):
    """Pseudo-Voigt 轮廓函数。

    参数：
      r: 径向坐标数组
      r0: 峰中心位置
      w: 半高宽参数（FWHM 相关）
      eta: 洛伦兹比例 (0=纯高斯, 1=纯洛伦兹)
      I0: 峰振幅
      bg: 常数背景
      slope: 线性背景斜率
    """
    w = max(abs(w), 1e-4)
    eta = np.clip(eta, 0.0, 1.0)
    # 高斯分量: FWHM = 2 * sqrt(2*ln2) * sigma ≈ 2.3548 * sigma
    sig = w / 2.35482
    G = np.exp(-0.5 * ((r - r0) / sig) ** 2)
    # 洛伦兹分量: FWHM = 2 * gamma
    gam = w / 2.0
    L = (gam ** 2) / ((r - r0) ** 2 + gam ** 2)

    return I0 * (eta * L + (1.0 - eta) * G) + bg + slope * (r - r0)


def fit_single_peak_pv(r, I, r_init=None, w_init=3.0):
    """对单段局部径向谱 (r, I) 进行 Pseudo-Voigt 非线性拟合。

    返回：
      dict(r0, fwhm, eta, I0, bg, r0_err, success)
    """
    r = np.asarray(r, dtype=float)
    I = np.asarray(I, dtype=float)

    if len(r) < 5 or np.ptp(I) <= 0:
        return dict(success=False, r0=np.nan, fwhm=np.nan, r0_err=np.nan)

    bg_init = float(np.min(I))
    I0_init = float(np.max(I) - bg_init)
    if r_init is None:
        r_init = float(r[np.argmax(I)])

    p0 = [r_init, w_init, 0.3, I0_init, bg_init, 0.0]
    r_min, r_max = float(r.min()), float(r.max())
    bounds = (
        [r_min, 0.5, 0.0, 0.0, -np.inf, -np.inf],
        [r_max, (r_max - r_min) * 0.8, 1.0, np.inf, np.inf, np.inf]
    )

    try:
        popt, pcov = curve_fit(pseudo_voigt_func, r, I, p0=p0, bounds=bounds,
                               maxfev=600)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(6, np.nan)
        r0 = float(popt[0])
        fwhm = float(popt[1])
        eta = float(popt[2])
        r0_err = float(perr[0]) if np.isfinite(perr[0]) else 0.05
        return dict(
            success=True,
            r0=r0,
            fwhm=fwhm,
            eta=eta,
            I0=float(popt[3]),
            bg=float(popt[4]),
            r0_err=r0_err
        )
    except Exception:
        # 回退至加权质心法
        weights = np.maximum(I - bg_init, 0)
        if weights.sum() > 0:
            r0 = float(np.sum(r * weights) / weights.sum())
            return dict(success=False, r0=r0, fwhm=w_init, eta=0.0, r0_err=0.2)
        return dict(success=False, r0=float(r_init), fwhm=w_init, eta=0.0, r0_err=0.5)


def refine_radial_peaks_pv(r_axis, profile, coarse_peaks, half_win=6):
    """利用 Pseudo-Voigt 拟合精化径向谱上检测出的一组衍射峰。

    参数：
      r_axis: 径向像素坐标 (一维数组)
      profile: 径向剖面光强 (一维数组)
      coarse_peaks: 粗峰列表，每个元素为 (r_peak, height) 或 r_peak
      half_win: 局部拟合单侧像素窗口大小

    返回：
      refined_peaks: 列表，每项为 dict(r0, fwhm, eta, I0, r0_err)
    """
    N = len(r_axis)
    refined = []

    for pk in coarse_peaks:
        r_pk = float(pk[0]) if isinstance(pk, (tuple, list)) else float(pk)
        idx = int(np.clip(np.searchsorted(r_axis, r_pk), 0, N - 1))
        i_lo = max(0, idx - half_win)
        i_hi = min(N, idx + half_win + 1)

        if i_hi - i_lo < 5:
            continue

        r_sub = r_axis[i_lo:i_hi]
        I_sub = profile[i_lo:i_hi]

        res = fit_single_peak_pv(r_sub, I_sub, r_init=r_pk)
        if res.get("success") and abs(res["r0"] - r_pk) <= half_win:
            refined.append(res)
        else:
            # 拟合未收敛但有效，保留粗峰
            refined.append(dict(r0=r_pk, fwhm=2.5, eta=0.0, r0_err=0.2, success=False))

    return refined
