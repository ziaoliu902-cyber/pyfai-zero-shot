# -*- coding: utf-8 -*-
"""bootstrap_uncertainty.py：块自助误差棒（B 路线 Phase 4a，论文素材）。

所有核心结果升级为 值±σ(stat)：
- D（圆解）：配对环为重采样单元（环=观测），重拟合 (cx,cy,D) → σ(D)。
  注意 σ 是纯统计部分；系统差（双线残迹/织构/畸变插值）不包含——
  probe 同时打印 σ_stat vs |ΔGT|，论文如实报告两者比例（诚实口径）。
- 倾角：扇区为重采样块，跨环同步重采样（保留环间相关结构），重拟合
  5 参（与 fit_tilt 同模型/同损失/同界，p0=点估计）→ σ(rot1/rot2)。
  控制点取自 fit_tilt(data_out=)——与生产拟合严格同一数据，另有点估计
  复现自检（同数据重拟合须回到 p0，防 bootstrap 装置本身有 bug）。

用法: python bootstrap_uncertainty.py ceria|alumina [dbg]
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio

from solver_core import solve, ring_radii, to_canvas, _r_model, _lam_A
from poni_out import fit_tilt, make_pyfai_fwd
from torchfai.phases import phase_qs, PHASES

MODE = sys.argv[1] if len(sys.argv) > 1 else "ceria"
DBG = "dbg" in sys.argv[2:]
B_D = 400          # D bootstrap 次数（lm 拟合微秒级）
B_TILT = 300       # 倾角 bootstrap 次数（含 pyFAI 前向，较贵）
SEED = 20261031


# ------------------------------------------------------------ 通用件
def d_bootstrap(canvas, cx, cy, D_px, q_list, lam, px_m, B=B_D, seed=SEED):
    """环级重采样 → σ(D)。在解点重配对——容差复刻 solve 择优规则
    （配对多者胜，平手取紧容差：远距离档生产的最终轮常是 3%），
    以配对环为单元有放回重采样，重拟合 (cx,cy,D)。
    配对环<4 时统计上不可辨识（dof≤0）→ 返回 ('few', n)。
    返回 (σ_mm, ci_lo, ci_hi, n_pairs) / ('few', n) / None。"""
    lamA = _lam_A(lam)
    qs = np.asarray(q_list, float)
    peaks, *_ = ring_radii(canvas, cx, cy)
    best = None
    for frac in (0.03, 0.01):            # 平手（同配对数）保留紧容差
        pairs = []
        for q in qs:
            rm = _r_model(np.array([q]), lamA, D_px)[0]
            tol = max(1.5, frac * rm)
            c = [(abs(rp - rm), rp) for rp, _ in peaks if abs(rp - rm) < tol]
            if c:
                pairs.append((q, min(c)[1]))
        if best is None or len(pairs) > len(best):
            best = pairs
    pairs = best
    if len(pairs) < 4:
        return ("few", len(pairs))
    qv = np.array([p[0] for p in pairs])
    rv = np.array([p[1] for p in pairs])
    rng = np.random.default_rng(seed)
    n = len(pairs)
    Ds = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        qq, rr = qv[idx], rv[idx]
        if len(np.unique(qq)) < 3:
            continue

        def resid(p):
            return _r_model(qq, lamA, p[2]) - rr

        s = least_squares(resid, [cx, cy, D_px], method="lm")
        Ds.append(s.x[2])
    if len(Ds) < B * 0.6:
        return None
    Ds = np.array(Ds) * px_m * 1000
    return (float(np.std(Ds)), float(np.percentile(Ds, 2.5)),
            float(np.percentile(Ds, 97.5)), n)


def tilt_bootstrap(do, px1, px2, fwd, B=B_TILT, seed=SEED, dbg=False):
    """扇区块重采样 → σ(rot1/rot2/dist)。块=扇区（跨环同步重采样，
    点按块重复计入=自然权重）；每 draw 过 fit_tilt 同款门（≥3 环 ≥30 点）。
    返回 dict 或 None。"""
    d1, d2, tth, w = do["d1"], do["d2"], do["tth"], do["w"]
    sec, qidx, nsec = do["sec"], do["qidx"], do["nsec"]
    p0 = list(do["p0"])
    dist0, p1_0, p2_0 = do["dist0"], do["poni1_0"], do["poni2_0"]
    lo = [dist0 - 0.03, p1_0 - 40 * px1, p2_0 - 40 * px2, -0.06, -0.06]
    hi = [dist0 + 0.03, p1_0 + 40 * px1, p2_0 + 40 * px2, 0.06, 0.06]

    def fit5(idx):
        def resid(p):
            return (fwd(p[0], p[1], p[2], p[3], p[4],
                        d1[idx], d2[idx]) - tth[idx]) * w[idx]
        return least_squares(resid, p0, loss="soft_l1", f_scale=1.5e-3,
                             x_scale=[0.01, 0.005, 0.005, 0.01, 0.01],
                             bounds=(lo, hi)).x

    # 装置自检：全数据重拟合须复现点估计（1e-6 rad 容差=数值抖动级别；
    # 防的是"重拟合掉进另一盆地"这类装置 bug）
    x = fit5(np.arange(len(d1)))
    if np.abs(np.asarray(x) - np.asarray(p0)).max() > 1e-6:
        print(f"  !! 点估计复现失败 maxΔ={np.abs(np.asarray(x)-np.asarray(p0)).max():.2e}")
        return None
    rng = np.random.default_rng(seed)
    sectors = np.arange(nsec)
    by_sec = {s: np.nonzero(sec == s)[0] for s in sectors if (sec == s).any()}
    live = np.array(sorted(by_sec))
    r1s, r2s, dmm = [], [], []
    skip = 0
    for _ in range(B):
        chosen = rng.choice(live, size=len(live), replace=True)
        idx = np.concatenate([by_sec[s] for s in chosen])
        if len(np.unique(qidx[idx])) < 3 or len(idx) < 30:
            skip += 1
            continue
        xx = fit5(idx)
        r1s.append(xx[3]); r2s.append(xx[4]); dmm.append(xx[0] * 1000)
    if len(r1s) < B * 0.8:
        if dbg:
            print(f"  !! 有效draw不足 {len(r1s)}/{B} (skip={skip})")
        return None
    r1s = np.array(r1s); r2s = np.array(r2s); dmm = np.array(dmm)
    return dict(
        s_r1=float(r1s.std() * 1e3), s_r2=float(r2s.std() * 1e3),
        s_D=float(dmm.std()),
        ci_r1=np.percentile(r1s, [2.5, 97.5]) * 1e3,
        ci_r2=np.percentile(r2s, [2.5, 97.5]) * 1e3,
        ndraw=len(r1s), npts=len(d1), nrings=len(np.unique(qidx)),
        nsec_live=len(live))


# ---------------------------------------------------------------- ceria
if MODE in ("ceria", "ceriac"):
    import pyFAI

    Q = phase_qs("CeO2")[:8]
    if MODE == "ceriac":
        # 经典优先路径（1031 消融发现）：to_canvas + 几何中心 + 全域投票，
        # 无 NN。精度 0.2/0.3mm（vs 混合路径 4.1/8.0）。
        print("== ceria[classic-first]: D 环级 bootstrap + 倾角扇区 bootstrap ==")
        from solver_core import to_canvas as _tc
    else:
        import torch

        from real_refine_v6 import load_px_nn, to_nn_input_wide
        from torchfai.dataset import TorchFAIDataset as _T
        from torchfai.dataset import norm_to_param

        LAM = 0.972386
        PXEFF = 172e-6 / (521.0 / 2527)
        ds = _T(os.path.join(ROOT, "data", "train_v4w3"), "val")
        consts = ds.norm_consts
        m1, dev, _ = load_px_nn(r"runs\v4s1\best.pt"); m1.eval()
        print("== ceria：D 环级 bootstrap + 倾角扇区 bootstrap ==")

    print(f"{'帧':>5}{'D_fit':>9}{'±σ':>7}{'95%CI(mm)':>22}"
          f"{'|ΔGT|':>8}{'GT∈CI':>7}{'配对':>5}"
          f"{'σr1':>7}{'σr2':>7}{'倾角pt/环/扇':>14}")
    sig_all, err_all = [], []
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(
            ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = p.dist * 1000
        if MODE == "ceriac":
            canvas, s, cx0, cy0 = _tc(img)
            pxm = 172e-6 * s
            LAMc = 0.972386
            peaks = ring_radii(canvas, cx0, cy0)[0]
            sol = solve(peaks, cx0, cy0, 1.0 / pxm, Q, LAMc, pxm,
                        d_range=(0.05, 3.0), rms_floor=1.8)
            wide, pxm, LAM = canvas, pxm, LAMc
        else:
            wide, k, _ = to_nn_input_wide(img)
            x = torch.from_numpy(
                np.log1p(wide[None, None].astype(np.float32))).to(dev)
            with torch.no_grad():
                mu, _ = m1(x, torch.tensor([LAM], device=dev),
                           torch.tensor([k], device=dev))
            p6 = norm_to_param(mu.double().cpu().numpy(), consts, k)[0]
            Hm, Wm = wide.shape
            cx0 = p6[2] / PXEFF * (Wm / 490.0)
            cy0 = p6[1] / PXEFF * (Hm / 521.0)
            D_px0 = p6[0] / PXEFF * (Hm / 521.0)
            pxm = PXEFF * 521.0 / Hm
            peaks = ring_radii(wide, cx0, cy0)[0]
            sol = solve(peaks, cx0, cy0, D_px0, Q, LAM, pxm, dbg=DBG,
                        D_ref_m=p6[0], d_range=(0.35, 2.8), rms_floor=1.8)
        if sol is None:
            print(f"{d:>5}  圆解失败")
            continue
        cx, cy, D_px = sol[0], sol[1], sol[2]
        Dmm = D_px * pxm * 1000
        bs = d_bootstrap(wide, cx, cy, D_px, Q, LAM, pxm)
        # 倾角 bootstrap（生产同款参数）
        do = {}
        tf = fit_tilt(wide, cx, cy, D_px, Q, LAM, pxm, pxm, sfull=1,
                      nsec=36, mass_rel=0.10, bump=1.5, data_out=do,
                      dbg=DBG)
        tb = None
        if tf:
            tb = tilt_bootstrap(do, pxm, pxm, make_pyfai_fwd(pxm, pxm, None),
                                dbg=DBG)
        e = abs(Dmm - gt)
        s_r1 = f"{tb['s_r1']:6.2f}" if tb else "   n/a"
        s_r2 = f"{tb['s_r2']:6.2f}" if tb else "   n/a"
        ns = (f"{tb['npts']}pt/{tb['nrings']}环/{tb['nsec_live']}扇"
              if tb else "-")
        if bs is None:
            print(f"{d:>5}{Dmm:9.1f}{'':>7}{'':>22}{e:8.1f}{'':>7}{'':>5}"
                  f"{s_r1:>7}{s_r2:>7}{ns:>14}  bootstrap失败")
            continue
        if bs[0] == "few":
            print(f"{d:>5}{Dmm:9.1f}{'  n/a':>7}"
                  f"{'(环=' + str(bs[1]) + ', dof≤0)':>22}{e:8.1f}"
                  f"{'':>7}{bs[1]:>5}{s_r1:>7}{s_r2:>7}{ns:>14}")
            err_all.append(e)
            continue
        s_, lo_, hi_, np_ = bs
        in_ci = "✓" if lo_ <= gt <= hi_ else "✗"
        sig_all.append(s_); err_all.append(e)
        print(f"{d:>5}{Dmm:9.1f}{s_:7.2f}"
              f"  [{lo_:7.1f},{hi_:7.1f}]{e:8.1f}{in_ci:>7}{np_:>5}"
              f"{s_r1:>7}{s_r2:>7}{ns:>14}")
    if sig_all:
        print(f"\n聚合: mean σ_stat(D)={np.mean(sig_all):.2f}mm"
              f"(有σ的{len(sig_all)}档)  "
              f"mean|ΔGT|={np.mean(err_all):.2f}mm(全{len(err_all)}档)  "
              f"σ/实际≈{np.mean(sig_all)/np.mean(err_all):.2f}"
              f"  （GT∈95%CI 全✓=统计域·区间有效；多✗=系统差主导·"
              f"区间过窄，见上表）")

# -------------------------------------------------------------- alumina
elif MODE == "alumina":
    import glob

    D_IR = sorted(glob.glob(r"data\real\alumina\*.edf"))
    txt = dict(l.split(":", 1) for l in open(
        glob.glob(r"data\real\alumina\*.poni")[0])
        if ":" in l and not l.startswith("#"))
    gt = float(txt["Distance"]) * 1000
    lam = float(txt["Wavelength"])
    px1, px2 = float(txt["PixelSize1"]), float(txt["PixelSize2"])
    rot1_gt, rot2_gt = float(txt["Rot1"]), float(txt["Rot2"])
    cons = "Al2O3"
    Q = phase_qs(cons)
    IR = PHASES[cons][1]

    print("== alumina：逐帧 D bootstrap（环级）+ 平均帧倾角 bootstrap（扇区）==")
    print(f"GT: D={gt:.2f}mm rot=({rot1_gt*1e3:+.2f},{rot2_gt*1e3:+.2f})mrad")
    print(f"{'帧':>18}{'D_mm':>8}{'±σ':>7}{'95%CI(mm)':>22}"
          f"{'GT∈CI':>7}{'配对':>5}")
    sig_all = []
    Dmm_all = []
    t0 = time.time()
    for f in D_IR:
        img = fabio.open(f).data.astype(np.float64)
        canvas, s, cx0, cy0 = to_canvas(img)
        px_m = (px1 + px2) / 2 * s
        peaks, prof, bin_px, good = ring_radii(canvas, cx0, cy0)
        sol = solve(peaks, cx0, cy0, 1.0 / px_m, Q, lam, px_m, I_rel=IR,
                    prof_ctx=(prof, bin_px, good), canvas=canvas)
        if sol is None or sol[3] < 5:
            print(f"{os.path.basename(f):>18}  失败")
            continue
        cx, cy, D_px = sol[0], sol[1], sol[2]
        Dmm = D_px * px_m * 1000
        bs = d_bootstrap(canvas, cx, cy, D_px, Q, lam, px_m)
        if bs is None:
            print(f"{os.path.basename(f):>18}{Dmm:8.1f}  bootstrap失败")
            continue
        if bs[0] == "few":
            print(f"{os.path.basename(f):>18}{Dmm:8.1f}"
                  f"{'n/a(环' + str(bs[1]) + ')':>16}")
            continue
        s_, lo_, hi_, np_ = bs
        in_ci = "✓" if lo_ <= gt <= hi_ else "✗"
        sig_all.append(s_); Dmm_all.append(Dmm)
        print(f"{os.path.basename(f):>18}{Dmm:8.1f}{s_:7.2f}"
              f"  [{lo_:7.1f},{hi_:7.1f}]{in_ci:>7}{np_:>5}")
    sig_all = np.array(sig_all); Dmm_all = np.array(Dmm_all)
    med = float(np.median(Dmm_all))
    print(f"\n逐帧: mean σ_stat={sig_all.mean():.2f}mm  "
          f"帧间MAD={np.median(np.abs(Dmm_all-med)):.2f}mm  "
          f"中位D−GT={med-gt:+.2f}mm  "
          f"(σ_stat≈帧内统计, |中位−GT|=系统差主导, 两者比≈"
          f"{abs(med-gt)/max(sig_all.mean(),1e-9):.1f}×)")
    print(f"逐帧耗时 {time.time()-t0:.0f}s")

    # ---- 平均帧倾角 bootstrap（spline 前向，生产同款）----
    spline = os.path.join(ROOT, "data", "real", "alumina",
                          "distorsion_2x2.spline")
    imgm = np.mean([fabio.open(f).data for f in D_IR], 0).astype(np.float64)
    canvas, s, cx0, cy0 = to_canvas(imgm)
    px_m = (px1 + px2) / 2 * s
    peaks, prof, bin_px, good = ring_radii(canvas, cx0, cy0)
    sol = solve(peaks, cx0, cy0, 1.0 / px_m, Q, lam, px_m, I_rel=IR,
                prof_ctx=(prof, bin_px, good), canvas=canvas)
    assert sol, "平均帧圆解失败"
    cx, cy, D_px = sol[0], sol[1], sol[2]
    do = {}
    tf = fit_tilt(canvas, cx, cy, D_px, Q, lam, px1 * s, px2 * s, sfull=s,
                  fwd=make_pyfai_fwd(px1, px2, spline), nsec=36,
                  mass_rel=0.10, bump=1.5, data_out=do, dbg=DBG)
    if tf:
        print(f"\n平均帧倾角点估计: rot=({tf['rot1']*1e3:+.2f},"
              f"{tf['rot2']*1e3:+.2f})mrad (GT {rot1_gt*1e3:+.2f},"
              f"{rot2_gt*1e3:+.2f})  D={tf['dist']*1000:.2f}mm")
        tb = tilt_bootstrap(do, px1 * s, px2 * s,
                            make_pyfai_fwd(px1, px2, spline), dbg=DBG)
        if tb:
            print(f"扇区bootstrap({tb['ndraw']}draw, {tb['npts']}pt/"
                  f"{tb['nrings']}环/{tb['nsec_live']}活扇):")
            print(f"  rot1 = {tf['rot1']*1e3:+7.2f} ± {tb['s_r1']:.2f} mrad"
                  f"  95%CI[{tb['ci_r1'][0]:+.2f},{tb['ci_r1'][1]:+.2f}]"
                  f"  (ΔGT {(tf['rot1']-rot1_gt)*1e3:+.2f})")
            print(f"  rot2 = {tf['rot2']*1e3:+7.2f} ± {tb['s_r2']:.2f} mrad"
                  f"  95%CI[{tb['ci_r2'][0]:+.2f},{tb['ci_r2'][1]:+.2f}]"
                  f"  (ΔGT {(tf['rot2']-rot2_gt)*1e3:+.2f})")
            print(f"  dist = {tf['dist']*1000:7.2f} ± {tb['s_D']:.2f} mm"
                  f"  (ΔGT {tf['dist']*1000-gt:+.2f})")
            print("  → σ 为扇区重采样统计；|ΔGT| 显著大于 σ 即系统差主导")
else:
    print(__doc__)
