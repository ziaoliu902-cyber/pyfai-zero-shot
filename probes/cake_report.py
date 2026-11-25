# -*- coding: utf-8 -*-
"""cake_report.py：2θ/q-χ 蛋糕展开图与 2x2 标定诊断体检图导出。"""
import matplotlib
import numpy as np
from matplotlib.figure import Figure

if "Microsoft YaHei" not in matplotlib.rcParams.get("font.sans-serif", []):
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
matplotlib.rcParams["axes.unicode_minus"] = False

from poni_out import _lam_A, fit_tilt, make_pyfai_fwd, tth_of
from solver_core import _r_model, ring_radii, to_canvas


def compute_cake(img, dist_m, poni1_m, poni2_m, rot1=0.0, rot2=0.0,
                 px1=172e-6, px2=None, lam_A=1.0, n_q=500, n_chi=180,
                 mask=None, spline=None):
    """纯 numpy 计算 (q, χ) 二维蛋糕展开图，返回 dict(cake, q_axis, chi_axis, tth_axis)。"""
    arr = np.asarray(img, dtype=np.float64)
    H, W = arr.shape
    px1_f = float(px1)
    px2_f = px1_f if px2 is None else float(px2)
    lam = _lam_A(lam_A)

    sub = 2 if max(H, W) > 2000 else 1
    ys = np.arange(0, H, sub, dtype=np.float64)[:, None]
    xs = np.arange(0, W, sub, dtype=np.float64)[None, :]
    vals = arr[::sub, ::sub]

    p1 = (ys + 0.5) * px1_f
    p2 = (xs + 0.5) * px2_f
    if spline is not None:
        try:
            fwd = make_pyfai_fwd(px1_f, px2_f, spline)
            tth = fwd(float(dist_m), float(poni1_m), float(poni2_m),
                      float(rot1), float(rot2), ys, xs)
        except Exception:
            tth = tth_of(p1, p2, float(dist_m), float(poni1_m), float(poni2_m),
                         float(rot1), float(rot2))
    else:
        tth = tth_of(p1, p2, float(dist_m), float(poni1_m), float(poni2_m),
                     float(rot1), float(rot2))
    q_pix = 4.0 * np.pi * np.sin(0.5 * tth) / lam
    chi = np.rad2deg(np.arctan2(p1 - float(poni1_m), p2 - float(poni2_m)))

    valid = np.isfinite(vals) & (vals < 1e9) & (vals >= 0.0) & np.isfinite(q_pix)
    if mask is not None and np.asarray(mask).shape == (H, W):
        valid &= (~np.asarray(mask, dtype=bool)[::sub, ::sub])

    if not valid.any():
        q_ax = np.linspace(0.1, 5.0, int(n_q))
        chi_ax = np.linspace(-180.0, 180.0, int(n_chi), endpoint=False) + 180.0 / n_chi
        return dict(
            cake=np.full((int(n_chi), int(n_q)), np.nan),
            q_axis=q_ax,
            chi_axis=chi_ax,
            tth_axis=2.0 * np.rad2deg(np.arcsin(np.clip(q_ax * lam / (4.0 * np.pi), 0.0, 0.9999))),
        )

    qv = q_pix[valid].ravel()
    cv = chi[valid].ravel()
    vv = vals[valid].ravel()
    q_min = float(np.percentile(qv, 0.5))
    q_max = float(np.percentile(qv, 99.5))
    if not np.isfinite(q_min) or q_max <= q_min:
        q_min, q_max = float(qv.min()), float(qv.max()) + 1e-6

    in_r = (qv >= q_min) & (qv <= q_max)
    qv, cv, vv = qv[in_r], cv[in_r], vv[in_r]

    nq, nc = int(n_q), int(n_chi)
    iq = np.clip(((qv - q_min) / max(q_max - q_min, 1e-12) * nq).astype(int), 0, nq - 1)
    ic = np.clip(((cv + 180.0) / 360.0 * nc).astype(int), 0, nc - 1)
    flat = ic * nq + iq
    cnt = np.bincount(flat, minlength=nc * nq).reshape(nc, nq)
    sums = np.bincount(flat, weights=vv, minlength=nc * nq).reshape(nc, nq)
    cake = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)

    q_ax = q_min + (np.arange(nq, dtype=np.float64) + 0.5) * (q_max - q_min) / nq
    chi_ax = -180.0 + (np.arange(nc, dtype=np.float64) + 0.5) * 360.0 / nc
    tth_ax = 2.0 * np.rad2deg(np.arcsin(np.clip(q_ax * lam / (4.0 * np.pi), 0.0, 0.9999)))
    return dict(cake=cake, q_axis=q_ax, chi_axis=chi_ax, tth_axis=tth_ax)


def draw_cake(ax, cake_dict, qs=None, title="q-χ 蛋糕展开图"):
    """在 matplotlib Axes 上绘制 log1p 蛋糕展开图并叠加理论环竖线。"""
    ax.clear()
    cake = cake_dict["cake"]
    q_ax = cake_dict["q_axis"]
    chi_ax = cake_dict["chi_axis"]
    disp = np.log1p(np.clip(cake, 0.0, None))
    fin = disp[np.isfinite(disp)]
    if fin.size > 0:
        vmin, vmax = np.percentile(fin, [5.0, 99.0])
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    ax.imshow(
        disp, origin="lower", aspect="auto",
        extent=[float(q_ax[0]), float(q_ax[-1]), float(chi_ax[0]), float(chi_ax[-1])],
        cmap="viridis", vmin=float(vmin), vmax=float(vmax),
    )
    if qs is not None:
        for q0 in np.asarray(qs, dtype=float):
            if float(q_ax[0]) <= q0 <= float(q_ax[-1]):
                ax.axvline(float(q0), color="#2ecc71", lw=0.9, ls="--", alpha=0.85)
    ax.set_xlabel(r"q (Å$^{-1}$)")
    ax.set_ylabel("χ (°)")
    ax.set_title(title, fontsize=9)


def _geom_from_res(res, px1_m, px2_m=None):
    sol = res["sol"]
    s = float(res.get("s", 1))
    px1_f = float(px1_m)
    px2_f = px1_f if px2_m is None else float(px2_m)
    if "cx" in res and "cy" in res and "D_px" in res and "tilt_on" in res:
        # gui_solver.SolveWorker 结果格式（cx/cy/D_px 为原生像素）
        tf = res.get("tf")
        if res.get("tilt_on") and isinstance(tf, dict):
            return (float(tf["dist"]), float(tf["poni1"]), float(tf["poni2"]),
                    float(tf["rot1"]), float(tf["rot2"]))
        return (
            float(res["D_px"]) * px1_f,
            (float(res["cy"]) + 0.5) * px1_f,
            (float(res["cx"]) + 0.5) * px2_f,
            0.0,
            0.0,
        )
    px_c = float(res["px_m"])
    tf = res.get("tilt_ref")
    rf5 = res.get("refine5")
    if isinstance(tf, dict) and tf.get("applied"):
        return (float(tf["dist"]), float(tf["poni1"]), float(tf["poni2"]),
                float(tf["rot1"]), float(tf["rot2"]))
    if isinstance(rf5, dict) and rf5.get("ok"):
        p = rf5["p"]
        return float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
    return (
        float(sol[2]) * px_c,
        (float(sol[1]) * s + 0.5) * px1_f,
        (float(sol[0]) * s + 0.5) * px2_f,
        0.0,
        0.0,
    )


def export_diagnostic_figure(out_path, img, res, px1_m, px2_m=None, mask=None, spline=None):
    """生成 2x2 标定体检图（2D 环叠图 / q-χ Cake / 1D 径向谱 / 参数 QA 卡片）并保存。"""
    from matplotlib.patches import Circle
    from torchfai.phases import phase_qs

    arr = np.asarray(img, dtype=np.float64)
    px1_f = float(px1_m)
    px2_f = px1_f if px2_m is None else float(px2_m)
    lam = _lam_A(float(res["lam"]))
    sol = res["sol"]
    s = float(res.get("s", 1))
    phase = res.get("phase") or res.get("ph")
    qs = phase_qs(phase) if phase else np.array([], dtype=float)
    dist_m, poni1_m, poni2_m, rot1, rot2 = _geom_from_res(res, px1_f, px2_f)

    canvas = res.get("canvas")
    if canvas is None:
        canvas, s, _, _ = to_canvas(arr)
    if spline is not None and len(qs) >= 4:
        try:
            fwd_sp = make_pyfai_fwd(px1_f, px2_f, spline)
            tf_sp = fit_tilt(canvas, float(sol[0]), float(sol[1]), float(sol[2]),
                             qs, lam, px1_f * s, px2_f * s, sfull=int(round(s)),
                             fwd=fwd_sp, nsec=36, win=6.0, mass_rel=0.10,
                             bump=1.5, suspect_tol=5e-3, min_gap=14.0)
            if tf_sp is not None and tf_sp["nrings"] >= 4 and tf_sp["rms5"] < 0.006:
                dist_m = float(tf_sp["dist"])
                poni1_m = float(tf_sp["poni1"])
                poni2_m = float(tf_sp["poni2"])
                rot1 = float(tf_sp["rot1"])
                rot2 = float(tf_sp["rot2"])
        except Exception:
            pass

    fig = Figure(figsize=(12, 9))
    ax_img = fig.add_subplot(2, 2, 1)
    ax_cake = fig.add_subplot(2, 2, 2)
    ax_1d = fig.add_subplot(2, 2, 3)
    ax_card = fig.add_subplot(2, 2, 4)

    # 1. 2D 衍射图 + 束心 + 理论环 + mask
    c_disp = np.log1p(np.clip(canvas, 0.0, None))
    fin = c_disp[np.isfinite(c_disp)]
    vlo, vhi = np.percentile(fin, [5.0, 99.2]) if fin.size else (0.0, 1.0)
    ax_img.imshow(c_disp, origin="upper", cmap="gray", vmin=float(vlo), vmax=float(vhi))
    if mask is not None and np.asarray(mask).shape == arr.shape:
        mc, _, _, _ = to_canvas(np.asarray(mask, dtype=np.float64), max_side=max(canvas.shape))
        ov = np.zeros((*mc.shape, 4), dtype=np.float64)
        m_b = mc > 0.25
        ov[m_b] = [1.0, 0.15, 0.15, 0.35]
        ax_img.imshow(ov, origin="upper")
    if "cx" in res and "cy" in res and "D_px" in res and "tilt_on" in res:
        cx_nat, cy_nat, D_nat = float(res["cx"]), float(res["cy"]), float(res["D_px"])
        cx_c, cy_c, D_c = cx_nat / max(s, 1.0), cy_nat / max(s, 1.0), D_nat / max(s, 1.0)
    else:
        cx_c, cy_c, D_c = float(sol[0]), float(sol[1]), float(sol[2])
        cx_nat, cy_nat = (cx_c + 0.5) * s - 0.5, (cy_c + 0.5) * s - 0.5
    ax_img.plot([cx_c], [cy_c], "r+", ms=12, mew=1.8)
    for r_c in _r_model(qs, lam, D_c):
        if 5.0 < r_c < 1.5 * max(canvas.shape):
            ax_img.add_patch(Circle((cx_c, cy_c), float(r_c), fill=False,
                                    ec="#2ecc71", lw=0.8, alpha=0.75))
    ax_img.set_title(f"2D 环叠图 ({phase or 'Custom'} | D={dist_m*1000:.2f} mm)", fontsize=9)

    # 2. q-χ Cake
    cake = compute_cake(arr, dist_m, poni1_m, poni2_m, rot1=rot1, rot2=rot2,
                        px1=px1_f, px2=px2_f, lam_A=lam, mask=mask, spline=spline)
    draw_cake(ax_cake, cake, qs=qs, title="q-χ 蛋糕展开图（环平直即无倾角残差）")

    # 3. 1D 径向谱（从已校正倾角与像素各向异性的 Cake 沿方位角均值导出）
    q_1d = cake["q_axis"]
    prof_1d = np.nanmean(cake["cake"], axis=0)
    valid_1d = np.isfinite(prof_1d)
    if valid_1d.any():
        ax_1d.plot(q_1d[valid_1d], np.log1p(np.clip(prof_1d[valid_1d], 0.0, None)),
                   color="#3498db", lw=1.0)
    for q0 in qs:
        if q_1d[0] <= q0 <= q_1d[-1]:
            ax_1d.axvline(float(q0), color="#e74c3c", lw=0.75, ls="--", alpha=0.7)
    ax_1d.set_xlabel(r"q (Å$^{-1}$)")
    ax_1d.set_ylabel("log1p(I)")
    ax_1d.set_title("1D 径向积分谱与理论峰位", fontsize=9)

    # 4. 参数 QA 摘要卡片
    ax_card.axis("off")
    conf = res.get("confidence") or {}
    m_v = res.get("m", int(sol[3]) if len(sol) > 3 else 0)
    rms_v = res.get("rms", float(sol[4]) if len(sol) > 4 else 0.0)
    ps_v = res.get("ps", float(sol[6]) if len(sol) > 6 else 0.0)
    nat_flag = res.get("native_2d", res.get("native", False))
    tf_obj = res.get("tilt_ref") or res.get("tf")
    tilt_ok = bool((isinstance(tf_obj, dict) and tf_obj.get("applied")) or res.get("tilt_on"))
    badge_txt = "".join(ch for ch in str(conf.get("badge", "")) if ord(ch) < 0x1F000).strip()
    lines = [
        f"Phase        : {phase}",
        f"Distance     : {dist_m * 1000:.3f} mm",
        f"Wavelength   : {lam:.5f} Å",
        f"Beam Center  : cx={cx_nat:.2f} px, cy={cy_nat:.2f} px (native)",
        f"PONI         : p1={poni1_m * 1000:.3f} mm, p2={poni2_m * 1000:.3f} mm",
        f"Tilt (rot1/2): {rot1 * 1e3:+.2f} / {rot2 * 1e3:+.2f} mrad",
        f"Matched m    : {m_v} | rms={rms_v:.3f} px | ps={ps_v:.3f}",
        f"Confidence   : {conf.get('tier', 'N/A')} ({badge_txt})",
        f"Az Coverage  : {res.get('az_coverage', 1.0) * 100:.0f}%",
        f"Refinement   : native_2d={nat_flag} | tilt_ok={tilt_ok}",
    ]
    ax_card.text(0.04, 0.92, "\n".join(lines), va="top", ha="left",
                 fontfamily=["Microsoft YaHei", "SimHei", "DejaVu Sans"], fontsize=9.5,
                 bbox=dict(boxstyle="round,pad=0.6", fc="#f8f9fa", ec="#bdc3c7"))
    ax_card.set_title("标定参数与质量评估摘要", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path
