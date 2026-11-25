# -*- coding: utf-8 -*-
"""auto_mask.py：自动坏点 + 束挡阴影 + 模块缝隙组合遮罩生成。"""
import numpy as np
from scipy.ndimage import label, maximum_filter, median_filter

from solver_core import auto_gap_mask


def detect_bad_pixels(img, hot_z=8.0):
    """检测无效像素（NaN/Inf/负值/哨兵 >=1e9）以及 3x3 孤立亮点（热像素）。"""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    invalid = (~np.isfinite(arr)) | (arr < 0.0) | (arr >= 1e9)
    valid = ~invalid
    if valid.sum() < 16:
        return invalid.copy()
    clean = arr.copy()
    if invalid.any():
        clean[invalid] = float(np.median(arr[valid]))
    med = median_filter(clean, size=3, mode="nearest")
    fp = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    nb_max = maximum_filter(clean, footprint=fp, mode="nearest")
    resid = clean - med
    r_valid = resid[valid]
    r_med = float(np.median(r_valid))
    mad = float(np.median(np.abs(r_valid - r_med)))
    sigma = max(1.4826 * mad, 1.0)
    hot = (
        valid
        & (clean > 4.0 * np.maximum(nb_max, 0.0) + max(15.0 * sigma, 30.0))
        & (resid > r_med + max(float(hot_z), 12.0) * sigma)
    )
    return invalid | hot


def detect_beamstop_shadow(img, cx=None, cy=None, dark_frac=0.25,
                           max_area_frac=0.15):
    """检测靠近束流中心的暗区连通域（束流挡盘/挡杆阴影）。"""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    H, W = arr.shape
    cx_u = (W - 1) / 2.0 if cx is None else float(cx)
    cy_u = (H - 1) / 2.0 if cy is None else float(cy)
    valid = np.isfinite(arr) & (arr >= 0.0) & (arr < 1e9)
    if valid.sum() < 100:
        return np.zeros(arr.shape, dtype=bool)
    vals = arr[valid]
    p_dark = float(np.percentile(vals, 0.1))
    p10 = float(np.percentile(vals, 10.0))
    p50 = float(np.percentile(vals, 50.0))
    if p50 <= p_dark:
        return np.zeros(arr.shape, dtype=bool)
    thr = min(p10, p_dark + float(dark_frac) * (p50 - p_dark))
    dark = valid & (arr <= thr)
    if not dark.any():
        return np.zeros(arr.shape, dtype=bool)

    lbl, n_comp = label(dark, structure=np.ones((3, 3), dtype=int))
    if n_comp == 0:
        return np.zeros(arr.shape, dtype=bool)

    r_center = 0.08 * float(min(H, W))
    y0 = max(0, int(np.floor(cy_u - r_center)))
    y1 = min(H, int(np.ceil(cy_u + r_center)) + 1)
    x0 = max(0, int(np.floor(cx_u - r_center)))
    x1 = min(W, int(np.ceil(cx_u + r_center)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    center_win = ((yy - cy_u) ** 2 + (xx - cx_u) ** 2) <= (r_center ** 2)
    touch = np.unique(lbl[y0:y1, x0:x1][center_win])
    touch = touch[touch > 0]
    if len(touch) == 0:
        return np.zeros(arr.shape, dtype=bool)

    counts = np.bincount(lbl.ravel())
    max_area = float(max_area_frac) * float(arr.size)
    keep_ids = [int(cid) for cid in touch if 0 < counts[int(cid)] <= max_area]
    if not keep_ids:
        return np.zeros(arr.shape, dtype=bool)
    return np.isin(lbl, keep_ids)


def build_auto_mask(img, cx=None, cy=None, include_gaps=True,
                    include_bad=True, include_beamstop=True):
    """组合模块缝隙 + 坏点/热像素 + 束挡阴影遮罩，返回 (mask_bool, stats_dict)。"""
    arr = np.asarray(img)
    shape = arr.shape[:2]
    m_gap = np.zeros(shape, dtype=bool)
    m_bad = np.zeros(shape, dtype=bool)
    m_bs = np.zeros(shape, dtype=bool)
    if include_gaps:
        try:
            g = auto_gap_mask(arr)
            if g is not None and g.shape == shape:
                m_gap = np.asarray(g, dtype=bool)
        except Exception:
            pass
    if include_bad:
        m_bad = detect_bad_pixels(arr)
    if include_beamstop:
        m_bs = detect_beamstop_shadow(arr, cx=cx, cy=cy)
    comb = m_gap | m_bad | m_bs
    n_tot = int(comb.sum())
    stats = dict(
        n_gap=int(m_gap.sum()),
        n_bad=int(m_bad.sum()),
        n_beamstop=int(m_bs.sum()),
        n_total=n_tot,
        frac=float(n_tot / max(arr.size, 1)),
    )
    return comb, stats
