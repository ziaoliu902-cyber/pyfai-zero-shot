# -*- coding: utf-8 -*-
"""poni_out.py：倾角拟合（rot1/rot2）+ .poni 标定文件输出（B 路线 Phase 3）。

前向模型（与 pyFAI calc_pos_zyx 逐点同构，__main__ 数值对拍）：
  v = (p1−poni1, p2−poni2, dist)，P = R2·R1·v（pyFAI 符号约定：R1/R2 左手系）
  tth = arccos(P_z / |P|)；p1=慢轴(行)，p2=快轴(列)，单位米。

物理要点：
- rot3≡0：绕光轴自转不改变 q-map（rot3_invariance.py 实测 max|dq|~1e-7），
  1D 曲线严格不变 → 固定 0（GT alumina poni rot3=1e-8 佐证）。
- 倾角信息在 r(χ) 一阶谐波，幅度 ∝ tan(tth) 随环增大；束心偏移是绝对
  位移不随半径缩放 → 跨环可分，多环控制点 5 参 LSQ 可辨识。
- 倾角≠0 时 PONI ≠ 环心（束流穿透点），差 O(tilt·D)（alumina ~6 画布px）；
  控制点参数化（绝对像素坐标固定，poni 是自由参）自动吸收，
  与 pyFAI GeometryRefinement 同构。

格式：镜像 data/real/alumina GT poni（旧版 PixelSize1/2 裸键，
pyFAI 2025.12 实测可读回）。
"""
import numpy as np
from scipy.optimize import least_squares

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _lam_A(lam):
    return lam * 1e10 if lam < 1e-6 else float(lam)


def tth_of(p1, p2, dist, poni1, poni2, rot1, rot2):
    """pyFAI 同构前向：像素物理坐标(m) → 2θ(rad)。rot3≡0。"""
    v1 = p1 - poni1
    v2 = p2 - poni2
    ca, sa = np.cos(rot1), np.sin(rot1)
    cb, sb = np.cos(rot2), np.sin(rot2)
    x1 = v1                                   # R1 不动分量0
    y1 = ca * v2 + sa * dist
    z1 = -sa * v2 + ca * dist
    x2 = cb * x1 - sb * z1                    # R2 作用后（模长须用旋转后的全分量）
    z2 = sb * x1 + cb * z1
    n2 = x2 * x2 + y1 * y1 + z2 * z2
    return np.arccos(np.clip(z2 / np.sqrt(np.maximum(n2, 1e-30)), -1, 1))


def sector_peaks(canvas, cx, cy, radii, nsec=12, win=6.0, mass_rel=0.12,
                 bump=1.25):
    """逐环×扇区径向峰位（控制点源）。
    每扇区 1px bin 径向直方图（权重=clip(I−bg,0)）→ argmax+抛物线内插。
    质量门：扇区质量 < mass_rel×环内中位质量 → 弃（织构暗区/出视野）；
    凸起门：峰高 > bump×直方图均值（防平坦噪声 argmax）。
    出视野环弧自然只产出存在像素的扇区（截断弧仍是有用控制点）。
    返回 (ring_idx, chi, r_px, mass) 列表。"""
    H, W = canvas.shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    r = np.hypot(yy - cy, xx - cx)
    chi = np.arctan2(yy - cy, xx - cx)
    bg = float(np.median(canvas))
    w_img = np.clip(canvas - bg, 0, None)
    nb = int(round(2 * win))
    out = []
    for k, r0 in enumerate(np.atleast_1d(radii)):
        sel = np.abs(r - r0) < win
        if sel.sum() < 300:
            continue
        rr = r[sel]
        cc = chi[sel]
        ww = w_img[sel]
        sec = ((cc + np.pi) / (2 * np.pi) * nsec).astype(int) % nsec
        mass = np.bincount(sec, weights=ww, minlength=nsec)
        m_med = float(np.median(mass[mass > 0])) if (mass > 0).any() else 0.0
        if m_med <= 0:
            continue
        lo = r0 - win
        for s in range(nsec):
            if mass[s] < mass_rel * m_med:
                continue
            m = sec == s
            bidx = np.clip(((rr[m] - lo) / (2 * win) * nb)
                           .astype(int), 0, nb - 1)
            prof = np.bincount(bidx, weights=ww[m], minlength=nb)
            pk = int(prof.argmax())
            if pk <= 0 or pk >= nb - 1:
                continue
            if prof[pk] < bump * prof.mean():
                continue
            y0, y1, y2 = prof[pk - 1], prof[pk], prof[pk + 1]
            d = 0.5 * (y0 - y2) / max(y0 - 2 * y1 + y2, 1e-12)
            rp = lo + (pk + 0.5 + np.clip(d, -1, 1)) * (2 * win / nb)
            chi_c = (s + 0.5) / nsec * 2 * np.pi - np.pi
            out.append((k, chi_c, float(rp), float(mass[s])))
    return out


def make_pyfai_fwd(px1=None, px2=None, spline=None):
    """构造 fwd(dist, poni1, poni2, rot1, rot2, d1full, d2full) → tth(rad)。

    spline 给定（数据集自带探测器畸变，如 Frelon distorsion_2x2.spline）：
    用 pyFAI Geometry 前向（含逐像素畸变校正），与 GT 标定口径一致；
    否则用自研 tth_of（已对拍 pyFAI 至 1e-14，无 pyFAI 运行时依赖）。
    d1full/d2full = 全分辨率像素索引（物理坐标 (d+0.5)·px 由内部处理）。"""
    if spline is None:
        def fwd(dist, poni1, poni2, rot1, rot2, d1, d2):
            return tth_of((np.asarray(d1) + 0.5) * px1,
                          (np.asarray(d2) + 0.5) * px2,
                          dist, poni1, poni2, rot1, rot2)
        return fwd
    import pyFAI.geometry
    from pyFAI.detectors import Detector

    try:
        det = Detector(pixel1=px1, pixel2=px2, splinefile=spline)
    except TypeError:                    # 旧版 pyFAI 参数名
        det = Detector(pixel1=px1, pixel2=px2, splineFile=spline)
    g = pyFAI.geometry.Geometry(dist=0.1, wavelength=1e-10)
    g.detector = det

    def fwd(dist, poni1, poni2, rot1, rot2, d1, d2):
        g.dist = dist
        g.poni1 = poni1
        g.poni2 = poni2
        g.rot1 = rot1
        g.rot2 = rot2
        g.rot3 = 0.0
        return np.asarray(g.tth(np.asarray(d1, float), np.asarray(d2, float)),
                          dtype=float)
    return fwd


def fit_tilt(canvas, cx, cy, D_px, qs, lam, px1, px2, sfull=1, fwd=None,
             nsec=18, win=6.0, mass_rel=0.12, bump=1.5, dbg=False,
             data_out=None):
    """圆解 (cx, cy, D_px 画布px) → 控制点 5 参 LSQ。

    px1/px2 = 画布两轴物理尺寸（m，含池化因子；各向异性进模型）。
    sfull = 全分辨率/画布池化因子（fwd 需要全分辨率像素索引）。
    fwd = make_pyfai_fwd(...)（spline 探测器）；None → 自研 tth_of。
    返回 dict(dist, poni1, poni2, rot1, rot2, rms5, rms3, imp, npts,
    nrings, tilt_deg) 或 None。rms5/rms3=含/不含倾角的 tth 残差 rms(rad)，
    imp = 相对改善。PONI 为全分辨率米制（画布坐标 (i+0.5)·px 直接换算）。
    data_out: 给 dict 时回填最终控制点（d1,d2,tth,w,sec,qidx,nsec,p0,
    dist0,poni1_0,poni2_0）——bootstrap_uncertainty.py 用于扇区重采样，
    与生产拟合严格同数据。不影响正常返回。
    """
    lam = _lam_A(lam)
    qs = np.asarray(qs, float)
    tt = 2 * np.arcsin(np.clip(qs * lam / (4 * np.pi), 0, 0.999))
    radii = D_px * np.tan(tt)
    H, W = canvas.shape
    vis = (radii > 40) & (radii < 0.98 * min(H, W))
    # —— 双线污染两段防御：未分辨双线(<14px≈2×FWHM)的融合峰位偏移由
    # 织构决定、I_rel 不可预测（alumina 实测偏 -15~-19mrad 恒定）。
    # ① 干净环（两侧间距均≥14px）先拟合 rot=0 三参锚定模型；
    # ② 嫌疑环按 |环中位残差|≤5mrad 逐个数据收编。2026-10-31 alumina。
    gap_ok = np.ones(len(qs), bool)
    for k in range(1, len(qs)):
        if abs(radii[k] - radii[k - 1]) < 14.0:
            gap_ok[k] = gap_ok[k - 1] = False
    base = np.where(vis & gap_ok)[0]
    suspect = np.where(vis & ~gap_ok)[0]
    if len(base) < 2:
        return None
    pts = sector_peaks(canvas, cx, cy, radii[base], nsec=nsec, win=win,
                       mass_rel=mass_rel, bump=bump)
    rings = {k for k, _, _, _ in pts}
    if len(rings) < 2 or len(pts) < 10:
        if dbg:
            print(f"    [dbg] 倾角: 干净环不足 {len(pts)}pt/{len(rings)}环")
        return None
    if len(suspect):
        sp = sector_peaks(canvas, cx, cy, radii[suspect], nsec=nsec, win=win,
                          mass_rel=mass_rel, bump=bump)
    else:
        sp = []
    ri = np.array([p[0] for p in pts])
    ch = np.array([p[1] for p in pts])
    rp = np.array([p[2] for p in pts])
    ms = np.array([p[3] for p in pts])
    rows = cy + rp * np.sin(ch)
    cols = cx + rp * np.cos(ch)
    p1 = (rows + 0.5) * px1
    p2 = (cols + 0.5) * px2
    d1f = (rows + 0.5) * sfull - 0.5      # 全分辨率像素索引
    d2f = (cols + 0.5) * sfull - 0.5
    tth_t = tt[base[ri]]
    wgt = np.sqrt(ms)
    wgt = wgt / wgt.mean()
    if fwd is None:
        # d1f/d2f 是全分辨率索引，fwd 内部必须用原生像素（px1/px2 是画布
        # 像素 = 原生×sfull）。sfull=1 时等价；sfull>1 曾把坐标放大 sfull
        # 倍导致倾角拟合撞边界（2026-10-31 ceria to_canvas 修复）。
        fwd = make_pyfai_fwd(px1 / sfull, px2 / sfull, None)

    dist0 = D_px * 0.5 * (px1 + px2)
    poni1_0 = (cy + 0.5) * px1
    poni2_0 = (cx + 0.5) * px2

    def model(p, dd1, dd2):
        if len(p) == 5:
            return fwd(p[0], p[1], p[2], p[3], p[4], dd1, dd2)
        return fwd(p[0], p[1], p[2], 0.0, 0.0, dd1, dd2)

    def lsq(p0, free_rot, dd1, dd2, tt_, w_):
        def resid(p):
            return (model(p, dd1, dd2) - tt_) * w_
        if free_rot:
            s = least_squares(resid, p0, loss="soft_l1", f_scale=1.5e-3,
                              x_scale=[0.01, 0.005, 0.005, 0.01, 0.01],
                              bounds=([dist0 - 0.03, poni1_0 - 40 * px1,
                                       poni2_0 - 40 * px2, -0.06, -0.06],
                                      [dist0 + 0.03, poni1_0 + 40 * px1,
                                       poni2_0 + 40 * px2, 0.06, 0.06]))
            return s.x, s.fun / w_
        s = least_squares(resid, p0[:3], loss="soft_l1", f_scale=1.5e-3,
                          x_scale=[0.01, 0.005, 0.005],
                          bounds=([dist0 - 0.03, poni1_0 - 40 * px1,
                                   poni2_0 - 40 * px2],
                                  [dist0 + 0.03, poni1_0 + 40 * px1,
                                   poni2_0 + 40 * px2]))
        return np.array([s.x[0], s.x[1], s.x[2], 0.0, 0.0]), s.fun / w_

    # ① 干净环锚定（rot=0）
    x3, _ = lsq([dist0, poni1_0, poni2_0], False, d1f, d2f, tth_t, wgt)

    # ② 嫌疑环数据收编
    admit = []
    if sp:
        sri = np.array([q[0] for q in sp])
        sch = np.array([q[1] for q in sp])
        srp = np.array([q[2] for q in sp])
        sms = np.array([q[3] for q in sp])
        srows = cy + srp * np.sin(sch)
        scols = cx + srp * np.cos(sch)
        sd1 = (srows + 0.5) * sfull - 0.5
        sd2 = (scols + 0.5) * sfull - 0.5
        stt = tt[suspect[sri]]
        for k in np.unique(sri):
            m = sri == k
            if m.sum() < 3:
                continue
            med = float(np.median(model(x3, sd1[m], sd2[m]) - stt[m]))
            if abs(med) <= 5e-3:
                admit.append(k)
            elif dbg:
                print(f"    [dbg] 倾角: 弃嫌疑环 r={radii[suspect[k]]:.0f}px"
                      f"（偏 {med*1e3:+.1f}mrad，双线融合）")
        if admit:
            m = np.isin(sri, admit)
            d1f = np.concatenate([d1f, sd1[m]])
            d2f = np.concatenate([d2f, sd2[m]])
            tth_t = np.concatenate([tth_t, stt[m]])
            w2 = np.sqrt(sms[m])
            w2 = w2 / w2.mean()
            wgt = np.concatenate([wgt, w2])
            wgt = wgt / wgt.mean()
            n_admit_rings = len(admit)
        else:
            n_admit_rings = 0
    else:
        n_admit_rings = 0
    n_rings_fin = len(rings) + n_admit_rings
    n_pts_fin = len(tth_t)
    if n_rings_fin < 3 or n_pts_fin < 30:
        if dbg:
            print(f"    [dbg] 倾角: 收编后不足 {n_pts_fin}pt/{n_rings_fin}环")
        return None

    # rot=0 三参基线 + 终拟合（放开 rot1/rot2，rot3≡0）
    x3, res3 = lsq([dist0, poni1_0, poni2_0], False, d1f, d2f, tth_t, wgt)
    x5, res5 = lsq(list(x3[:3]) + [0.0, 0.0], True, d1f, d2f, tth_t, wgt)
    rms5 = float(np.sqrt(np.mean(res5 ** 2)))
    rms3 = float(np.sqrt(np.mean(res3 ** 2)))
    imp = 1.0 - rms5 / max(rms3, 1e-12)
    dist, poni1, poni2, rot1, rot2 = [float(v) for v in x5]
    npts = int(n_pts_fin)
    nrings = int(n_rings_fin)
    if dbg:
        print(f"    [dbg] 倾角: {npts}pt/{nrings}环 "
              f"rms3={rms3*1e3:.2f} rms5={rms5*1e3:.2f}mrad "
              f"改善{imp*100:.0f}% rot=({rot1*1e3:+.1f},{rot2*1e3:+.1f})mrad "
              f"tilt={np.hypot(rot1, rot2)*180/np.pi:.3f}°")
    if data_out is not None:
        sec_b = ((ch + np.pi) / (2 * np.pi) * nsec).astype(int) % nsec
        qidx_b = base[ri]
        if len(sp) and admit:
            mm = np.isin(sri, admit)
            sec_s = (((sch + np.pi) / (2 * np.pi) * nsec)
                     .astype(int) % nsec)
            sec_f = np.concatenate([sec_b, sec_s[mm]])
            qidx_f = np.concatenate([qidx_b, suspect[sri[mm]]])
        else:
            sec_f, qidx_f = sec_b, qidx_b
        data_out.update(d1=d1f.copy(), d2=d2f.copy(), tth=tth_t.copy(),
                        w=wgt.copy(), sec=sec_f, qidx=qidx_f, nsec=nsec,
                        p0=(dist, poni1, poni2, rot1, rot2), dist0=dist0,
                        poni1_0=poni1_0, poni2_0=poni2_0)
    return dict(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2,
                rot3=0.0, rms5=rms5, rms3=rms3, imp=imp, npts=npts,
                nrings=nrings, tilt_deg=float(np.hypot(rot1, rot2)
                                              * 180 / np.pi))


def write_poni(path, dist, poni1, poni2, rot1, rot2, lam, px1, px2,
               rot3=0.0, note="", spline=None):
    """写 .poni（镜像 GT 旧版格式）+ pyFAI 回读自检。"""
    import time

    lines = [
        "# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis",
        f"# Auto-calibrated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
        "probes/solver_core.py + poni_out.py (B route)",
    ]
    if note:
        lines.append(f"# {note}")
    if spline:
        lines.append(f"SplineFile: {spline}")
        lines.append("# (spline taken from reference; NOT modeled in fit)")
    lines += [
        f"PixelSize1: {px1:.12g}",
        f"PixelSize2: {px2:.12g}",
        f"Distance: {dist:.12g}",
        f"Poni1: {poni1:.12g}",
        f"Poni2: {poni2:.12g}",
        f"Rot1: {rot1:.12g}",
        f"Rot2: {rot2:.12g}",
        f"Rot3: {rot3:.12g}",
        f"Wavelength: {lam:.12g}",
        "",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    # 回读自检（pyFAI 可用即验，硬失败；12g 精度 → 1e-11 m 级）
    try:
        import pyFAI
        g = pyFAI.load(path)
        chk = [abs(g.dist - dist), abs(g.poni1 - poni1), abs(g.poni2 - poni2),
               abs(g.rot1 - rot1), abs(g.rot2 - rot2)]
        assert max(chk) < 1e-11, f"poni 回读不一致: {chk}"
    except ImportError:
        pass
    return path


# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import pyFAI.geometry

    print("== 1. 前向模型 vs pyFAI tthArray 数值对拍 ==")
    rng = np.random.default_rng(42)
    H, W = 48, 44
    worst = 0.0
    for t in range(3):
        dist = 10 ** rng.uniform(np.log10(0.08), np.log10(0.6))
        poni1 = rng.uniform(0.3, 0.7) * H * 5e-5
        poni2 = rng.uniform(0.3, 0.7) * W * 5e-5
        rot1 = rng.uniform(-0.03, 0.03)
        rot2 = rng.uniform(-0.03, 0.03)
        px1, px2 = 1.0336e-4, 1.0253e-4
        g = pyFAI.geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2,
                                    rot1=rot1, rot2=rot2, rot3=0.0,
                                    pixel1=px1, pixel2=px2,
                                    wavelength=1e-10)
        yy, xx = np.mgrid[0:H, 0:W]
        ref = g.tth(yy.ravel(), xx.ravel())        # d1/d2=像素索引
        mine = tth_of((yy + 0.5) * px1, (xx + 0.5) * px2, dist, poni1, poni2,
                      rot1, rot2)
        worst = max(worst, float(np.abs(ref - mine.ravel()).max()))
        print(f"  trial{t}: max|Δtth|={np.abs(ref-mine.ravel()).max():.2e} rad")
    assert worst < 1e-9, "前向模型与 pyFAI 不一致！"
    print(f"  → 一致 (worst={worst:.1e})\n")

    print("== 2. alumina 单帧冒烟（帧0000，指定相 Al2O3）==")
    import fabio
    from solver_core import solve, to_canvas, ring_radii
    from torchfai.phases import phase_qs, PHASES

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    AL = os.path.join(ROOT, "data", "real", "alumina")
    txt = dict(l.split(":", 1) for l in open(
        os.path.join(AL, "al2o3_00_max_51_frames.poni"))
        if ":" in l and not l.startswith("#"))
    lam = float(txt["Wavelength"])
    px1f, px2f = float(txt["PixelSize1"]), float(txt["PixelSize2"])
    gD, gr1, gr2 = (float(txt["Distance"]), float(txt["Rot1"]),
                    float(txt["Rot2"]))
    gp1, gp2 = float(txt["Poni1"]), float(txt["Poni2"])
    spline = os.path.join(AL, "distorsion_2x2.spline")
    img = fabio.open(os.path.join(AL, "al2o3_0000.edf")).data.astype(np.float64)
    canvas, s, cx0, cy0 = to_canvas(img)
    Q = phase_qs("Al2O3")
    peaks, prof, bp, gd = ring_radii(canvas, cx0, cy0)
    sol = solve(peaks, cx0, cy0, 1.0 / ((px1f + px2f) / 2 * s), Q, lam,
                (px1f + px2f) / 2 * s, I_rel=PHASES["Al2O3"][1],
                prof_ctx=(prof, bp, gd), canvas=canvas)
    assert sol, "圆解失败"
    cx, cy, D, m, rms = sol[:5]
    print(f"  圆解: D={D*(px1f+px2f)/2*s*1000:.2f}mm 环数={m} rms={rms:.2f}px")
    fwd = make_pyfai_fwd(px1f, px2f, spline)
    tf = fit_tilt(canvas, cx, cy, D, Q, lam, px1f * s, px2f * s, sfull=s,
                  fwd=fwd, dbg=True)
    if tf:
        print(f"  倾角: rot=({tf['rot1']*1e3:+.2f},{tf['rot2']*1e3:+.2f})mrad "
              f"(GT {gr1*1e3:+.2f},{gr2*1e3:+.2f}) "
              f"Δ=({(tf['rot1']-gr1)*1e3:+.2f},{(tf['rot2']-gr2)*1e3:+.2f})")
        print(f"  D: {tf['dist']*1000:.2f}mm (GT {gD*1000:.2f}, "
              f"Δ{(tf['dist']-gD)*1000:+.2f})  "
              f"poniΔ=({(tf['poni1']-gp1)/px1f:+.1f},"
              f"{(tf['poni2']-gp2)/px2f:+.1f})px")
        out = write_poni(os.path.join(ROOT, "probes", "results",
                                      "_smoke_alumina.poni"),
                         tf["dist"], tf["poni1"], tf["poni2"], tf["rot1"],
                         tf["rot2"], lam, px1f, px2f,
                         note="smoke frame al2o3_0000, phase Al2O3")
        print(f"  poni 已写+回读通过: {out}")
