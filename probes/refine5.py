# -*- coding: utf-8 -*-
"""五参数精化（可选最后一步）：在初值几何上做加权最小二乘，拟合
   (dist, poni1, poni2, rot1, rot2)，目标函数是"每个像素的 2θ 与其所属环理论
   2θ 之差"，像素按强度加权、按当前几何指派到最近的环，迭代数轮。

设计要点（都是踩过的坑）：
  1. **不能用窄环带取像素**：环带由初值几何决定，初值偏了就看不出偏差；
  2. **必须按强度加权**：否则背景像素把残差抹平；
  3. **必须迭代**：每轮用新几何重新指派像素；
  4. **必须给接受条件**：改善不足或位移过大就保留原解。
"""

import numpy as np

from geomfit import fit_geometry, tth_from_pixel


def _profile_peaks(tth, vals, nb=3000):
    edges = np.linspace(np.nanmin(tth), np.nanmax(tth), nb + 1)
    idx = np.clip(np.digitize(tth.ravel(), edges) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    sumi = np.bincount(idx, weights=vals.ravel(), minlength=nb)
    prof = sumi / np.maximum(cnt, 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    return prof, ctr


def refine_geometry5(img, D, cx, cy, qs, px, lam, n_iter=3, subsample=3,
                     max_shift_px=40.0, min_gain=0.15, tol=0.01):
    """返回 dict(p, gain, shift_px, cond, ok, before, after)。

    D: 米；cx, cy: 像素（图像坐标，x=列, y=行）；qs: 该相位的 q 列表（Å⁻¹）
    """
    H, W = img.shape
    ys, xs = np.mgrid[0:H:subsample, 0:W:subsample]
    vals = img[::subsample, ::subsample]
    tth_theo = 2 * np.arcsin(np.clip(lam * qs / (4 * np.pi), 0, 0.999))

    p = np.array([D, cy * px, cx * px, 0.0, 0.0], float)
    p_start = p.copy()

    good = False
    cond = np.nan
    n_rings_used = 0
    for _ in range(n_iter):
        tth = tth_from_pixel(xs, ys, *p, px)
        d = np.abs(tth[..., None] - tth_theo[None, None, :])
        near = d.argmin(axis=2)
        keep = d.min(axis=2) < tol
        w = np.clip(vals, 0, np.percentile(vals, 99.5)) * keep
        mx = w.max()
        if mx <= 0:
            return dict(p=p_start, gain=0.0, shift_px=0.0, cond=np.nan, ok=False,
                    before=float('nan'), after=float('nan'))
        w = w / mx
        m = w > 1e-3
        if m.sum() < 500:
            return dict(p=p_start, gain=0.0, shift_px=0.0, cond=np.nan, ok=False,
                    before=float('nan'), after=float('nan'))
        xy = np.column_stack([xs[m], ys[m]])
        obs = tth_theo[near[m]]
        n_rings_used = int(np.unique(obs).size)
        res = fit_geometry(xy, obs, p, px, weights=w[m])
        p = res["p"]
        cond = res["cond"]
        good = True

    if not good:
        return dict(p=p_start, gain=0.0, shift_px=0.0, cond=np.nan, ok=False,
                    before=float('nan'), after=float('nan'))

    before = _ring_fwhm(img, p_start, px, qs, lam)
    after = _ring_fwhm(img, p, px, qs, lam)
    shift = float(np.hypot(p[1] - p_start[1], p[2] - p_start[2]) / px)
    drel = abs(p[0] / p_start[0] - 1.0)
    gain = 0.0 if not np.isfinite(before) or before <= 0 else 1 - after / before

    # 局部最优检验（结果导向）：把圆心往四个方向各推 probe_px，
    # 若每个方向都变差，说明精修结果确实站在峰顶，而不是在斜坡上游荡。
    probe = 4.0
    worse = 0
    tested = 0
    if np.isfinite(after):
        for dx, dy in ((probe, 0), (-probe, 0), (0, probe), (0, -probe)):
            q = p.copy()
            q[2] += dx * px
            q[1] += dy * px
            f = _ring_fwhm(img, q, px, qs, lam)
            if np.isfinite(f):
                tested += 1
                if f > after:
                    worse += 1
    local_opt = (tested >= 3 and worse == tested)

    ok = (np.isfinite(after) and np.isfinite(before)
          and gain > min_gain                 # 必须明显变锐
          and shift < max_shift_px            # 兜底：不许离谱
          and cond < 1e4                      # 不许病态
          and n_rings_used >= 4               # 少于 4 条环：五参数欠定
          and drel < 0.02                     # 距离变化 <2%
          and local_opt)                      # 必须站在峰顶
    return dict(p=p, gain=gain, shift_px=shift, cond=cond, ok=ok,
                before=before, after=after, n_rings=n_rings_used, drel=drel,
                local_opt=local_opt, worse=worse, tested=tested)


def _ring_fwhm(img, p, px, qs, lam, npt=3000):
    """用给定几何做一次 1D 积分，返回前几条环的峰宽中位数（q 单位）。

    必须走"临时 poni → pyFAI.load"，否则 detector.shape 为空，
    splitBBox 会以 'Solid angle array size' 断言失败（踩过）。
    """
    import os
    import tempfile

    import pyFAI

    txt = (f"poni_version: 2.1\nDetector: Detector\n"
           f"Detector_config: {{\"pixel1\": {px}, \"pixel2\": {px}}}\n"
           f"Distance: {p[0]}\nPoni1: {p[1]}\nPoni2: {p[2]}\n"
           f"Rot1: {p[3]}\nRot2: {p[4]}\nRot3: 0.0\n"
           f"Wavelength: {lam * 1e-10}\n")
    fd, tmp = tempfile.mkstemp(suffix=".poni")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(txt)
    try:
        ai = pyFAI.load(tmp)
        q, I = ai.integrate1d(img, npt, unit="q_A^-1", method="bbox,splitpixel")
    except Exception:
        return float("nan")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    out = []
    for q0 in qs[:6]:
        sel = (q > q0 * 0.9) & (q < q0 * 1.1)
        if sel.sum() < 8:
            continue
        qs_, Is = q[sel], I[sel]
        base = float(np.median(Is))
        k = int(np.argmax(Is))
        if Is[k] < 1.5 * base:
            continue
        half = base + (Is[k] - base) / 2
        l = k
        while l > 0 and Is[l] > half:
            l -= 1
        r = k
        while r < len(Is) - 1 and Is[r] > half:
            r += 1
        out.append(qs_[r] - qs_[l])
    return float(np.median(out)) if out else float("nan")
