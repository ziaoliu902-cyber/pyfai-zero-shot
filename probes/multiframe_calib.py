# -*- coding: utf-8 -*-
"""multiframe_calib.py：多帧自动去暗场 + 离群剔除 + 叠加联合标定。"""
import os
from collections import Counter

import numpy as np

from batch_parallel import _load_img_headless, solve_batch_parallel
from poni_out import fit_tilt, make_pyfai_fwd, write_poni
from solver_core import auto_offset, solve_image
from torchfai.phases import phase_qs


def _is_dark_frame(img):
    """判定是否为纯暗场/无效空白帧（避免暗场混入联合平均拉低信噪比）。"""
    arr = np.asarray(img, dtype=np.float64)
    v = arr[np.isfinite(arr) & (arr < 1e9)]
    if v.size < 100:
        return True
    std = float(np.std(v))
    if std < 1e-6:
        return True
    p50 = float(np.percentile(v, 50.0))
    p99 = float(np.percentile(v, 99.0))
    dyn = p99 - p50
    if p99 > 20.0 and dyn < max(10.0, 0.08 * abs(p50)) and std < 10.0:
        return True
    return False


def calibrate_multiframe(paths, px1_m, lam=None, px2_m=None, phases=None,
                         spline=None, out_poni=None, n_workers=1, solve_kw=None):
    """对同工位多帧衍射图做自动暗场过滤、离群剔除、叠图和五参倾角联合标定。"""
    path_list = list(paths)
    if not path_list:
        raise ValueError("paths 不能为空")

    px1_f = float(px1_m)
    px2_f = px1_f if px2_m is None else float(px2_m)

    dark_files = []
    valid_paths = []
    imgs_by_path = {}
    for p in path_list:
        raw = _load_img_headless(p)
        if _is_dark_frame(raw):
            dark_files.append(p)
        else:
            valid_paths.append(p)
            imgs_by_path[p] = raw

    if not valid_paths:
        return dict(
            ok=False, error="全部输入均为暗场或无效帧",
            n_total=len(path_list), n_dark=len(dark_files),
            n_kept=0, n_outliers=0, dark_files=dark_files,
            kept_files=[], outlier_files=[], per_frame=[], poni_path=None,
        )

    per_frame = solve_batch_parallel(
        valid_paths, px1_f, lam=lam, px2_m=px2_f, phases=phases,
        n_workers=n_workers, solve_kw=solve_kw,
    )
    solved = [r for r in per_frame if r.get("ok")]
    if not solved:
        return dict(
            ok=False, error="非暗场帧均未能成功求解",
            n_total=len(path_list), n_dark=len(dark_files),
            n_kept=0, n_outliers=0, dark_files=dark_files,
            kept_files=[], outlier_files=[], per_frame=per_frame, poni_path=None,
        )

    # 主物相与离群帧判定（基于中位数与 MAD 剔除跑偏离群帧）
    ph_counts = Counter(r["phase"] for r in solved if r.get("phase"))
    dom_phase = phases[0] if (phases and len(phases) == 1) else (
        ph_counts.most_common(1)[0][0] if ph_counts else solved[0].get("phase"))

    d_vals = np.array([r["D_mm"] for r in solved], dtype=np.float64)
    med_d = float(np.median(d_vals))
    mad_d = float(np.median(np.abs(d_vals - med_d)))
    tol_d = 3.0 * max(1.4826 * mad_d, 0.5)

    kept_recs = []
    outlier_files = []
    for r in solved:
        if abs(r["D_mm"] - med_d) <= tol_d:
            kept_recs.append(r)
        else:
            outlier_files.append(r["path"])
    if not kept_recs:
        kept_recs = solved
        outlier_files = []
    for r in per_frame:
        if not r.get("ok"):
            outlier_files.append(r["path"])

    kept_files = [r["path"] for r in kept_recs]
    img_stack = np.mean(
        [auto_offset(imgs_by_path[p]) for p in kept_files], axis=0
    ).astype(np.float64)

    # 确定叠加图使用的相与波长
    lam_use = lam if lam is not None else float(np.median([r["lam_A"] for r in kept_recs]))
    ph_use = [dom_phase] if dom_phase else phases
    kw_use = dict(solve_kw) if solve_kw else dict(cascade=True, auto_center=False, refine_2d=False)
    res_mean = solve_image(
        img_stack, px1_f, px2_m=px2_f, lam=lam_use, phases=ph_use,
        solve_kw=kw_use,
    )
    if not res_mean:
        return dict(
            ok=False, error="联合叠加图求解失败",
            n_total=len(path_list), n_dark=len(dark_files),
            n_kept=len(kept_files), n_outliers=len(outlier_files),
            dark_files=dark_files, kept_files=kept_files,
            outlier_files=outlier_files, per_frame=per_frame, poni_path=None,
        )

    sol = res_mean["sol"]
    canvas, s = res_mean["canvas"], float(res_mean["s"])
    lam_fin = float(res_mean["lam"])
    phase_fin = res_mean.get("phase") or dom_phase
    qs = phase_qs(phase_fin)

    # 先用 poni_out.fit_tilt 在叠加图上联合拟合 5 参（含可选 spline），失败则退回 res_mean 结果
    fwd = make_pyfai_fwd(px1_f, px2_f, spline) if spline else None
    tf = fit_tilt(
        canvas, float(sol[0]), float(sol[1]), float(sol[2]), qs, lam_fin,
        px1_f * s, px2_f * s, sfull=s, fwd=fwd, nsec=36,
        mass_rel=0.10, bump=1.5, suspect_tol=2e-3, min_gap=14.0,
    )
    if tf is not None:
        dist_m = float(tf["dist"])
        poni1_m = float(tf["poni1"])
        poni2_m = float(tf["poni2"])
        rot1 = float(tf["rot1"])
        rot2 = float(tf["rot2"])
        tilt_deg = float(tf["tilt_deg"])
    else:
        tf_res = res_mean.get("tilt_ref")
        if isinstance(tf_res, dict) and tf_res.get("applied"):
            dist_m = float(tf_res["dist"])
            poni1_m = float(tf_res["poni1"])
            poni2_m = float(tf_res["poni2"])
            rot1 = float(tf_res["rot1"])
            rot2 = float(tf_res["rot2"])
            tilt_deg = float(tf_res["tilt_deg"])
        else:
            dist_m = float(sol[2]) * float(res_mean["px_m"])
            poni1_m = (float(sol[1]) * s + 0.5) * px1_f
            poni2_m = (float(sol[0]) * s + 0.5) * px2_f
            rot1, rot2, tilt_deg = 0.0, 0.0, 0.0

    poni_written = None
    if out_poni:
        note = (f"multiframe joint ({len(kept_files)}/{len(path_list)} frames, "
                f"{len(dark_files)} dark, {len(outlier_files)} outliers)")
        poni_written = write_poni(
            out_poni, dist_m, poni1_m, poni2_m, rot1, rot2,
            lam_fin, px1_f, px2_f, spline=spline, note=note,
        )

    return dict(
        ok=True,
        phase=phase_fin,
        lam_A=lam_fin,
        dist_mm=dist_m * 1000.0,
        poni1_m=poni1_m,
        poni2_m=poni2_m,
        rot1=rot1,
        rot2=rot2,
        tilt_deg=tilt_deg,
        n_total=len(path_list),
        n_dark=len(dark_files),
        n_kept=len(kept_files),
        n_outliers=len(outlier_files),
        dark_files=dark_files,
        kept_files=kept_files,
        outlier_files=outlier_files,
        per_frame=per_frame,
        poni_path=poni_written,
    )
