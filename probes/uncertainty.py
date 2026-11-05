# -*- coding: utf-8 -*-
"""uncertainty.py：给"值"配"±"（不需要 GT 的经验估计）。

· 距离 D：环间一致性。用解出的 (相, D) 预测每条环半径，逐环 Kasa 圆拟合
  重解一个 D_i，σ_D = std(D_i)/√n（加权最小二乘的统计标准误）。ceria 实测
  ~0.05–0.2mm；环太少（<3）返回 0 表示"未估"。
· 倾角 rot1/rot2：fit_tilt 对扇区数的敏感度（nsec=18 vs 36 重拟合，差/√2）。
  ceria 实测 0.8–23 mrad，是"诚实的角度精度"（比 LSQ 协方差保守）。

用法：
    from uncertainty import dist_sigma, tilt_with_sigma
    sD_px = dist_sigma(img, cx_native, cy_native, D_native, qs, lam, px_m)
    tf, s1, s2 = tilt_with_sigma(canvas, cx_c, cy_c, D_c, qs, lam, pxm, pxm, s)
"""
import numpy as np

from poni_out import fit_tilt


def _kasa(px, py):
    A = np.stack([px, py, np.ones_like(px)], 1)
    b = px * px + py * py
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a0, b0, c0 = sol
    r2 = c0 + (a0 * a0 + b0 * b0) / 4.0
    if r2 <= 0:
        return None
    return a0 / 2.0, b0 / 2.0, float(np.sqrt(r2))


def dist_sigma(img, cx, cy, D_px, qs, lam, px_m, half=3.0, min_rings=3):
    """D 的 1σ（与 D_px 同单位：像素）。px_m=单个像素的米制尺寸。"""
    qs = np.asarray(qs, float)
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    lam_A = lam if lam > 1e-6 else lam * 1e10   # 兼容 Å / 米两种输入
    tt = 2.0 * np.arcsin(np.clip(lam_A * qs / (4 * np.pi), 0, 0.999))
    radii = D_px * np.tan(tt)
    r_edge = min(cx, cy, W - 1 - cx, H - 1 - cy)
    Ds = []
    for r0, t in zip(radii, tt):
        if r0 < 30 or r0 > 0.9 * r_edge or t <= 0:
            continue
        band = np.abs(np.hypot(xx - cx, yy - cy) - r0) < half
        iy, ix = np.nonzero(band)
        if iy.size < 200:
            continue
        v = img[iy, ix]
        base = float(np.percentile(v, 35))
        peak = float(np.percentile(v, 98))
        if peak <= base:
            continue
        keep = v > base + 0.35 * (peak - base)
        px_, py_ = ix[keep].astype(float), iy[keep].astype(float)
        if px_.size < 150:
            continue
        res = _kasa(px_, py_)
        if res is None:
            continue
        for _ in range(2):
            rr = np.abs(np.hypot(px_ - res[0], py_ - res[1]) - res[2])
            sg = rr.std()
            m = rr < 2.5 * max(sg, 0.5)
            if m.sum() < 150 or m.all():
                break
            px_, py_ = px_[m], py_[m]
            res = _kasa(px_, py_)
            if res is None:
                break
        if res is None:
            continue
        Ds.append(res[2] / np.tan(t))
    if len(Ds) < min_rings:
        return 0.0
    return float(np.std(Ds, ddof=1) / np.sqrt(len(Ds)))


def tilt_with_sigma(canvas, cx, cy, D_px, qs, lam, px1, px2, sfull=1,
                    nsec_lo=18, nsec_hi=36):
    """倾角 + 1σ。返回 (tf, sig_rot1, sig_rot2)；tf 可能为 None。"""
    a = fit_tilt(canvas, cx, cy, D_px, qs, lam, px1, px2, sfull=sfull,
                 nsec=nsec_lo, mass_rel=0.10, bump=1.5)
    b = fit_tilt(canvas, cx, cy, D_px, qs, lam, px1, px2, sfull=sfull,
                 nsec=nsec_hi, mass_rel=0.10, bump=1.5)
    if a is None and b is None:
        return None, 0.0, 0.0
    tf = b or a
    if a is not None and b is not None:
        s1 = abs(a["rot1"] - b["rot1"]) / np.sqrt(2.0)
        s2 = abs(a["rot2"] - b["rot2"]) / np.sqrt(2.0)
    else:
        s1 = s2 = 0.0
    return tf, float(s1), float(s2)


def ring_rms(img, cx, cy, D_px, qs, lam, half=3.0, min_rings=2):
    """几何自洽质检：各环圆拟合残差的中位（px，环"锐不锐/圆不圆"）。

    不依赖 GT——"环是不是圆的"。中心对、D 对，则每条环的像素都落在
    同一圆上，残差小；中心偏 10px，环被拉成椭圆，残差立刻上去。
    返回 (rms_px, n_rings)；环不足返回 (0.0, 0)。
    """
    qs = np.asarray(qs, float)
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    lam_A = lam if lam > 1e-6 else lam * 1e10
    tt = 2.0 * np.arcsin(np.clip(lam_A * qs / (4 * np.pi), 0, 0.999))
    radii = D_px * np.tan(tt)
    r_edge = min(cx, cy, W - 1 - cx, H - 1 - cy)
    sig = []
    for r0 in radii:
        if r0 < 30 or r0 > 0.9 * r_edge:
            continue
        band = np.abs(np.hypot(xx - cx, yy - cy) - r0) < half
        iy, ix = np.nonzero(band)
        if iy.size < 200:
            continue
        v = img[iy, ix]
        base = float(np.percentile(v, 35))
        peak = float(np.percentile(v, 98))
        if peak <= base:
            continue
        keep = v > base + 0.35 * (peak - base)
        px_, py_ = ix[keep].astype(float), iy[keep].astype(float)
        if px_.size < 150:
            continue
        res = _kasa(px_, py_)
        if res is None:
            continue
        for _ in range(2):
            rr = np.abs(np.hypot(px_ - res[0], py_ - res[1]) - res[2])
            sg = rr.std()
            m = rr < 2.5 * max(sg, 0.5)
            if m.sum() < 150 or m.all():
                break
            px_, py_ = px_[m], py_[m]
            res = _kasa(px_, py_)
            if res is None:
                break
        if res is None:
            continue
        sig.append(float(rr.std()))
    if len(sig) < min_rings:
        return 0.0, 0
    return float(np.median(sig)), len(sig)

