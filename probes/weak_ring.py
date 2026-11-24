# -*- coding: utf-8 -*-
"""weak_ring.py：单帧弱信号挖掘与物理强度似然模块

用于拯救强背底（空气散射/荧光/陡峭小角背景）或极少可见环导致的信息匮乏帧：
  1. snip_background(): 经典 LLS-SNIP 迭代削峰背底估计（免疫窄/高强衍射峰，
     且在急剧下降的陡背底上完整保留高角弱环）；
  2. matched_filter_score(): 全谱连续二维高斯模板匹配评分（不依赖硬阈值找峰）；
  3. intensity_likelihood_score(): 对数峰强相关性 + 缺失强环惩罚，在少环谐波
     歧义时用物理强度比例辅助消歧。
"""

import numpy as np


def snip_background(prof, max_half_win=24, decreasing=True):
    """LLS-SNIP（Log-Log-Sqrt 迭代削峰）一维径向背底估计。

    参数：
      prof: 一维径向光强谱
      max_half_win: 最大迭代半窗宽（bin 数，默认 24；约为典型峰宽的 2~3 倍）
      decreasing: True 时窗宽由大到小递减（抑制孤立宽/尖双峰；标准 X 射线做法）

    返回：
      与 prof 等长的背景曲线（float64 数组）
    """
    y0 = np.asarray(prof, dtype=np.float64)
    n = y0.size
    if n < 8:
        return y0.copy()

    finite = np.isfinite(y0)
    if not finite.any():
        return np.zeros_like(y0)

    vmin = float(np.min(y0[finite]))
    y = np.where(finite, np.maximum(y0 - vmin, 0.0), 0.0)

    # LLS 变换压缩动态范围，使尖锐强峰与平滑背景在同一尺度比较
    v = np.log(np.log(np.sqrt(y + 1.0) + 1.0) + 1.0)

    w_max = int(min(max(1, max_half_win), max(1, (n - 1) // 3)))
    widths = range(w_max, 0, -1) if decreasing else range(1, w_max + 1)
    for w in widths:
        avg = 0.5 * (v[:-2 * w] + v[2 * w:])
        v[w:-w] = np.minimum(v[w:-w], avg)

    # 逆 LLS 变换还原至原始强度尺度
    bg = (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0 + vmin
    bg = np.minimum(bg, y0)
    return np.where(finite, bg, 0.0)


def matched_filter_score(prof_sub, bin_px, qs, I_rel, lam, D_px,
                         sigma_px=2.5, rmax_use=None, min_r=25.0):
    """全谱连续模板匹配打分（不需要预先硬阈值找峰）。

    原理：根据理论 (q, I_rel, λ, D_px) 生成平滑多环高斯模板，并与去背底后的
    径向谱计算零均值 Pearson 相关系数，即使高角弱环信噪比低仍能贡献正相关。

    返回：
      float in [-1.0, 1.0]（有效理论环 < 2 时返回 0.0）
    """
    p = np.asarray(prof_sub, dtype=np.float64)
    n = p.size
    if n < 16 or bin_px <= 0 or D_px <= 0:
        return 0.0

    qs = np.asarray(qs, dtype=np.float64)
    I_rel = (np.ones(len(qs), dtype=np.float64) if I_rel is None
             else np.asarray(I_rel, dtype=np.float64))

    r_axis = (np.arange(n, dtype=np.float64) + 0.5) * float(bin_px)
    r_hi = float(r_axis[-1]) * 0.98 if rmax_use is None else float(rmax_use)
    mask = (r_axis >= float(min_r)) & (r_axis <= r_hi) & np.isfinite(p)
    if mask.sum() < 16:
        return 0.0

    lam_A = float(lam) * 1e10 if float(lam) < 1e-6 else float(lam)
    arg = qs * lam_A / (4.0 * np.pi)
    valid = (arg > 0.0) & (arg < 0.999)
    if not valid.any():
        return 0.0

    tt = 2.0 * np.arcsin(arg[valid])
    r_th = float(D_px) * np.tan(tt)
    weights = np.sqrt(np.maximum(I_rel[valid], 0.0))

    in_range = (r_th >= float(min_r)) & (r_th <= r_hi) & (weights > 0.0)
    r_in = r_th[in_range]
    w_in = weights[in_range]
    if r_in.size < 2:
        return 0.0

    sig = max(float(sigma_px), float(bin_px) * 1.2)
    r_sub = r_axis[mask]
    tmpl = np.zeros_like(r_sub)
    for rk, wk in zip(r_in, w_in):
        tmpl += wk * np.exp(-0.5 * ((r_sub - rk) / sig) ** 2)

    obs = p[mask]
    obs_c = obs - float(np.mean(obs))
    tmpl_c = tmpl - float(np.mean(tmpl))
    denom = float(np.sqrt(np.sum(obs_c ** 2) * np.sum(tmpl_c ** 2)))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.sum(obs_c * tmpl_c) / denom, -1.0, 1.0))


def intensity_likelihood_score(peaks_rp, qs, I_rel, lam, D_px,
                               tol_frac=0.02, min_r=25.0):
    """物理强度比例似然与缺失强环惩罚（辅助消除少环谐波歧义）。

    返回：
      dict(score, n_matched, n_missing_strong, icorr_log)
    """
    qs = np.asarray(qs, dtype=np.float64)
    I_rel = (np.ones(len(qs), dtype=np.float64) if I_rel is None
             else np.asarray(I_rel, dtype=np.float64))
    if not peaks_rp or len(qs) == 0 or D_px <= 0:
        return dict(score=0.0, n_matched=0, n_missing_strong=0, icorr_log=0.0)

    lam_A = float(lam) * 1e10 if float(lam) < 1e-6 else float(lam)
    arg = qs * lam_A / (4.0 * np.pi)
    valid = (arg > 0.0) & (arg < 0.999)
    if not valid.any():
        return dict(score=0.0, n_matched=0, n_missing_strong=0, icorr_log=0.0)

    tt = 2.0 * np.arcsin(arg[valid])
    r_th_all = float(D_px) * np.tan(tt)
    I_valid = np.maximum(I_rel[valid], 0.0)
    r_max_obs = max(float(rp) for rp, _ in peaks_rp) * 1.02

    # 全局最近 1-to-1 配对
    cand = []
    for qi, r_th in enumerate(r_th_all):
        tol = max(2.0, float(tol_frac) * r_th)
        for pi, (rp, h) in enumerate(peaks_rp):
            err = abs(float(rp) - float(r_th))
            if err <= tol:
                cand.append((err, pi, qi, float(h)))
    cand.sort()
    used_p, used_q, matched_I, matched_h = set(), set(), [], []
    for _err, pi, qi, h in cand:
        if pi in used_p or qi in used_q:
            continue
        used_p.add(pi)
        used_q.add(qi)
        matched_I.append(I_valid[qi])
        matched_h.append(max(h, 0.0))

    n_matched = len(matched_I)
    if n_matched >= 2:
        x = np.log1p(np.asarray(matched_I, dtype=np.float64))
        y = np.log1p(np.asarray(matched_h, dtype=np.float64))
        xc = x - float(np.mean(x))
        yc = y - float(np.mean(y))
        den = float(np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2)))
        icorr_log = float(np.clip(np.sum(xc * yc) / den, -1.0, 1.0)) if den > 1e-12 else 0.0
    else:
        icorr_log = 0.0

    # 观测半径范围内理论上强 (>=40% 最强环) 却完全没配上的环数
    i_max = float(np.max(I_valid)) if I_valid.size else 0.0
    n_missing_strong = 0
    for qi, (r_th, iv) in enumerate(zip(r_th_all, I_valid)):
        if qi not in used_q and min_r <= r_th <= r_max_obs and iv >= 0.40 * i_max:
            n_missing_strong += 1

    score = float(icorr_log - 0.25 * n_missing_strong)
    return dict(
        score=score,
        n_matched=int(n_matched),
        n_missing_strong=int(n_missing_strong),
        icorr_log=float(icorr_log),
    )


def azimuthal_consistency_score(canvas, cx, cy, qs, lam, D_px,
                                n_sec=24, half_px=3.0, min_r=25.0):
    """计算候选几何解在各独立方位扇区上的一致性得分。

    物理依据：真实多晶衍射环（即使存在一定织构）会在多个独立方位角扇区稳定高于
    局部背景；而单个强布拉格亮斑或偶发谐波伪解往往只在极少方位扇区碰撞出峰。

    返回：
      dict(score=float, n_rings=int)
    """
    img = np.asarray(canvas, dtype=np.float64)
    if img.ndim != 2 or D_px <= 0:
        return dict(score=0.0, n_rings=0)

    H, W = img.shape
    qs = np.asarray(qs, dtype=np.float64)
    lam_A = float(lam) * 1e10 if float(lam) < 1e-6 else float(lam)
    arg = qs * lam_A / (4.0 * np.pi)
    valid_q = (arg > 0.0) & (arg < 0.999)
    if not valid_q.any():
        return dict(score=0.0, n_rings=0)

    tt = 2.0 * np.arcsin(arg[valid_q])
    r_th_all = float(D_px) * np.tan(tt)

    yy, xx = np.mgrid[0:H, 0:W]
    dx = xx - float(cx)
    dy = yy - float(cy)
    r = np.hypot(dy, dx)
    r_max = float(r.max())
    r_in = r_th_all[(r_th_all >= float(min_r)) & (r_th_all <= 0.95 * r_max)]
    if r_in.size == 0:
        return dict(score=0.0, n_rings=0)

    n_sec = int(max(4, n_sec))
    sec = ((np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi) * n_sec).astype(np.int32) % n_sec

    fracs = []
    hp = float(half_px)
    for ri in r_in:
        dr = np.abs(r - float(ri))
        on_m = dr < hp
        bg_m = (dr >= hp + 2.0) & (dr <= hp + 7.0)
        if not on_m.any() or not bg_m.any():
            continue

        on_s = sec[on_m]
        on_v = img[on_m]
        on_cnt = np.bincount(on_s, minlength=n_sec)
        on_mean = np.bincount(on_s, weights=on_v, minlength=n_sec) / np.maximum(on_cnt, 1)

        bg_s = sec[bg_m]
        bg_v = img[bg_m]
        bg_cnt = np.bincount(bg_s, minlength=n_sec)
        bg_mean = np.bincount(bg_s, weights=bg_v, minlength=n_sec) / np.maximum(bg_cnt, 1)
        bg_sq = np.bincount(bg_s, weights=bg_v * bg_v, minlength=n_sec) / np.maximum(bg_cnt, 1)
        bg_std = np.sqrt(np.maximum(bg_sq - bg_mean * bg_mean, 0.0))

        valid_sec = (on_cnt >= 3) & (bg_cnt >= 6)
        if valid_sec.any():
            hit = valid_sec & (on_mean > bg_mean + 1.5 * bg_std)
            fracs.append(float(hit.sum()) / float(valid_sec.sum()))

    return dict(
        score=float(np.mean(fracs)) if fracs else 0.0,
        n_rings=int(len(fracs)),
    )
