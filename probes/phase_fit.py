# -*- coding: utf-8 -*-
"""phase_fit.py：多样品验收台。
用法: python phase_fit.py ceria|alumina|lab6 [dbg]

- ceria  : 回归——老路径(to_nn_input_wide+NN初值)必须复现七档通过
- alumina: 52帧 edf，GT=目录内 poni(2016)，λ=17.5keV，NN-free 全自动
- lab6   : 15帧 edf，λ=EDF头，NN-free；参考解=pyFAI精化.npt手工控制点
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio

from solver_core import (solve, solve_auto, solve_cascade, to_canvas,
                         canvas_of, solve_image, refine_geometry_2d)
from torchfai.phases import phase_qs

MODE = sys.argv[1] if len(sys.argv) > 1 else "ceria"
DBG = "dbg" in sys.argv[2:]
# 原生 2D 精化：ceria 默认开（交付 D 0.49→0.07mm），alumina 默认关（无增益且更慢）
R2D = ("r2d" in sys.argv[2:]) or (MODE == "ceria" and "nor2d" not in sys.argv[2:])


def acc(fit_mm, gt_mm):
    return fit_mm - gt_mm


# ---------------------------------------------------------------- ceria
if MODE == "ceria":
    import pyFAI

    from solver_core import solve_cascade, to_canvas

    LAM = 0.972386
    PX_NATIVE = 172e-6
    # 预注册同款 8 强环（phases.py 多的第 9 弱环 (422) I=8 会污染投票：
    # 其票散进中半径伪峰，挤出真簇——2026-10-31 回归教训）
    Q = phase_qs("CeO2")[:8]

    print(f"{'帧':>5}{'fitΔmm':>9}{'环数':>5}{'rms_px':>8}{'种子':>6}")
    es = []
    ROT = []
    _NN = {}

    def nn_prior(img, s):
        """wide 画布 NN 先验 → 全帧画布坐标（仅经典路径失败时调用）。

        NN 训在 521×490 裁剪画布（zoom f=521/H 后左侧裁 c0 列），其
        poni 输出是裁剪局部像素；换算：xf=p6[2]/px_native+c0/f，
        再池化到 to_canvas 的 s 倍画布。
        """
        import torch

        from real_refine_v6 import load_px_nn, to_nn_input_wide
        from torchfai.dataset import TorchFAIDataset as _T
        from torchfai.dataset import norm_to_param
        if "m" not in _NN:
            _NN["ds"] = _T(os.path.join(ROOT, "data", "train_v4w3"), "val")
            _NN["m"], _NN["dev"], _ = load_px_nn(r"runs\v4s1\best.pt")
            _NN["m"].eval()
        wide, k, _ = to_nn_input_wide(img)
        x = torch.from_numpy(
            np.log1p(wide[None, None].astype(np.float32))).to(_NN["dev"])
        with torch.no_grad():
            mu, _ = _NN["m"](
                x, torch.tensor([LAM], device=_NN["dev"]),
                torch.tensor([k], device=_NN["dev"]))
        p6 = norm_to_param(mu.double().cpu().numpy(),
                           _NN["ds"].norm_consts, k)[0]
        H, W = img.shape
        f = 521.0 / H
        c0 = (round(W * f) - 490) // 2
        xf = p6[2] / PX_NATIVE + c0 / f
        yf = p6[1] / PX_NATIVE
        return (xf + 0.5) / s - 0.5, (yf + 0.5) / s - 0.5, float(p6[0])

    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(
            ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = p.dist * 1000

        # ---- classic-first（R3 统一入口，2026-11-05）：solve_image 锁表
        # 模式 q_list=Q（预注册 8 强环）+ rms_floor=1.8，数值路径与旧直调
        # solve_cascade(to_canvas 520, Q, rms_floor=1.8) 完全一致；锁表
        # 模式无相鉴定风险，R2(c) 风险预门不参与（ceria s=5 免扫）----
        res = solve_image(img, PX_NATIVE, lam=LAM, q_list=Q,
                          solve_kw=dict(rms_floor=1.8), dbg=DBG)
        if res is not None:
            canvas, s = res["canvas"], res["s"]
            pxm = PX_NATIVE * s
            sol = res["sol"]
        else:
            # 经典路径两档都交不出解 → NN 先验兜底（跨画布映射），再走级联
            canvas, s, _, _ = canvas_of(img, "default")
            pxm = PX_NATIVE * s
            sol = None
        if sol is None:
            cx_nn, cy_nn, D_nn = nn_prior(img, s)
            if DBG:
                print(f"      经典失败 → NN 兜底 中心=({cx_nn:.1f},"
                      f"{cy_nn:.1f}) D={D_nn*1000:.1f}mm")
            sol = solve_cascade(canvas, LAM, pxm, Q, rms_floor=1.8,
                                cx_nn=cx_nn, cy_nn=cy_nn, D_nn_m=D_nn,
                                classic_first=True, dbg=DBG)
        if sol is None:
            print(f"{d:>5}  失败")
            es.append(999.0)
            continue
        cx, cy, D, m, rms = sol[:5]
        seed = sol[7] if len(sol) > 7 else "?"
        use2d = False
        ncx = ncy = nD = None
        if R2D:
            _o = refine_geometry_2d(img, cx * s, cy * s, D * s, Q, LAM,
                                    n_iter=3)
            if _o is not None:
                ncx, ncy, nD = _o[0], _o[1], _o[2]
                cx, cy, D = ncx / s, ncy / s, nD / s
                use2d = True
                seed += "+2D"
        e = D * pxm * 1000 - gt
        es.append(abs(e))

        # ---- 倾角精化 + poni 输出 ----
        from poni_out import fit_tilt, write_poni
        tf = fit_tilt(canvas, cx, cy, D, Q, LAM, pxm, pxm, sfull=s,
                      nsec=36, mass_rel=0.10, bump=1.5, dbg=DBG)
        gr1 = float(p.rot1)
        gr2 = float(p.rot2)
        if tf:
            # 混合口径（2026-11-03）：D/中心用原生 2D 精化（实测 0.05–0.1mm），
            # 倾角用 fit_tilt 的 rot。fit_tilt 会把 dist 重拟回 0.4mm 级，
            # 所以它的 dist 不再采用——只借 rot。
            # 混合口径：dist 用求解器（r2d 有则 r2d，实测 0.05–0.1mm），
            # poni/rot 用 fit_tilt（中心对 GT 4–17px，比求解器 17–31px 准）。
            # fit_tilt 自己重拟的 dist 会回到 0.4mm 级，故不采用。
            dist_w = D * pxm
            p1_w = tf["poni1"]
            p2_w = tf["poni2"]
            out = write_poni(
                os.path.join(ROOT, "probes", "results",
                             f"ceria_{d}.poni"),
                dist_w, p1_w, p2_w,
                tf["rot1"], tf["rot2"], LAM * 1e-10, PX_NATIVE, PX_NATIVE,
                note=f"ceria {d}mm, classic-first canvas ({seed})")
            print(f"{d:>5}{e:+9.1f}{m:>5}{rms:8.2f}{seed:>6}"
                  f"  rotΔ=({(tf['rot1']-gr1)*1e3:+5.1f},"
                  f"{(tf['rot2']-gr2)*1e3:+5.1f})mrad"
                  f" {tf['npts']}pt/{tf['nrings']}环"
                  f" DΔ={dist_w*1000-gt:+5.1f} → {out}")
            ROT.append(((tf['rot1']-gr1)*1e3, (tf['rot2']-gr2)*1e3))
        else:
            print(f"{d:>5}{e:+9.1f}{m:>5}{rms:8.2f}{seed:>6}"
                  f"  倾角拟合失败(输出圆解)")
            ROT.append(None)
    print(f"\n回归判据（均值<10 最差<15）: 平均{np.mean(es):.1f} "
          f"最差{max(es):.1f} → {'✓通过' if np.mean(es)<10 and max(es)<15 else '✗未过'}")
    rok = [r for r in ROT if r]
    if rok:
        a1 = np.array([r[0] for r in rok]); a2 = np.array([r[1] for r in rok])
        print(f"倾角 vs GT: rot1Δ均值{a1.mean():+.1f}mrad(范围{a1.min():+.1f}~"
              f"{a1.max():+.1f}) rot2Δ均值{a2.mean():+.1f}mrad(范围{a2.min():+.1f}"
              f"~{a2.max():+.1f}) 成功{len(rok)}/7")

# -------------------------------------------------------------- alumina
elif MODE == "alumina":
    import glob

    import pyFAI

    D_IR = sorted(glob.glob(r"data\real\alumina\*.edf"))
    txt = dict(l.split(":", 1) for l in open(
        glob.glob(r"data\real\alumina\*.poni")[0])
        if ":" in l and not l.startswith("#"))
    gt = float(txt["Distance"])
    lam = float(txt["Wavelength"])
    px1, px2 = float(txt["PixelSize1"]), float(txt["PixelSize2"])
    rot1, rot2 = float(txt["Rot1"]), float(txt["Rot2"])
    gt_cx = float(txt["Poni2"]) / px2
    gt_cy = float(txt["Poni1"]) / px1
    print(f"GT: D={gt*1000:.2f}mm λ={lam*1e10:.4f}Å "
          f"rot=({rot1:.4f},{rot2:.4f}) "
          f"px=({px1*1e6:.1f},{px2*1e6:.1f})µm "
          f"GT中心=({gt_cx:.0f},{gt_cy:.0f})px")
    print(f"{'帧':>18}{'相':>8}{'D_mm':>8}{'Δmm':>8}{'环数':>5}{'rms_px':>8}")
    # ---- 参考：自动识别票型（辅助信息；相由用户指定 Al2O3）----
    picks = []
    for f in D_IR[::10]:
        img = fabio.open(f).data.astype(np.float64)
        # R3 统一入口（solve_kw=旧 solve_auto 直调默认三关，行为不变）
        res = solve_image(img, (px1 + px2) / 2, lam=lam,
                          solve_kw=dict(cascade=False, auto_center=False,
                                        refine_2d=False))
        picks.append(res["phase"] if res else None)
    from collections import Counter
    print(f"自动识别票型(抽样): {dict(Counter(picks))}")
    cons = "Al2O3"          # 用户指定标样（pyFAI 工作流同款）
    from torchfai.phases import PHASES
    Q = phase_qs(cons)
    IR_rel = PHASES[cons][1]
    # ---- 指定相全帧精解 ----
    es = []
    Dmm = []
    OKF = []
    for f in D_IR:
        img = fabio.open(f).data.astype(np.float64)
        # R3 统一入口：锁相 Al2O3（phases 单相 → solve_cascade+库内
        # I_rel+rms_floor 3.5，与旧直调数值一致；auto_center/refine_2d
        # 保持旧关态）
        res = solve_image(img, (px1 + px2) / 2, lam=lam, phases=[cons],
                          solve_kw=dict(cascade=True, auto_center=False,
                                        refine_2d=False), dbg=DBG)
        sol = res["sol"] if res else None
        s = res["s"] if res else None
        name = os.path.basename(f)
        if sol is None or sol[3] < 5:
            print(f"{name:>18}{'失败':>8}")
            es.append(999.0)
            continue
        cx, cy, D, m, rms = sol[:5]
        px_m = (px1 + px2) / 2 * s
        if R2D:
            _o = refine_geometry_2d(img, cx * s, cy * s, D * s, Q, lam,
                                    n_iter=3)
            if _o is not None:
                cx, cy, D = _o[0] / s, _o[1] / s, _o[2] / s
        e = D * px_m * 1000 - gt * 1000
        es.append(abs(e))
        Dmm.append(D * px_m * 1000)
        OKF.append(os.path.basename(f))
        print(f"{name:>18}{cons:>8}{D*px_m*1000:8.1f}{e:+8.1f}{m:>5}{rms:8.2f}")
    ok = [x for x in es if x < 900]
    Dmm = [d for d, e in zip(Dmm, es) if e < 900]
    med = float(np.median(Dmm))
    mad = float(np.median(np.abs(np.array(Dmm) - med)))
    outl = [f for f, d in zip(OKF, Dmm) if abs(d - med) > 3 * mad]
    print(f"\nalumina[指定{cons}]: 平均|Δ|={np.mean(ok):.1f}mm "
          f"最差{max(ok):.1f}mm 成功{len(ok)}/{len(es)}")
    print(f"静态装置 QA: 中位D={med:.1f}mm (ΔGT={med-gt*1000:+.1f}mm) "
          f"MAD={mad:.2f}mm 离群帧(>3MAD)={len(outl)}"
          f"{[''] + [','.join(outl)] if outl else ''}")

    # ---- 平均帧倾角精化 + poni 输出（B 路线 Phase3）----
    # 单帧倾角不可靠（rot 散布 ±20mrad：外环被 Al2O3 双线梳污染 + 织构χ孔洞
    # → 倾角-poni 近简并）；几何静止 → 51 帧平均把峰位噪声 ↓7×。
    # GT 口径 = poni + distorsion_2x2.spline（Frelon 畸变），拟合前向含 spline。
    from poni_out import fit_tilt, make_pyfai_fwd, write_poni
    spline = os.path.join(ROOT, "data", "real", "alumina",
                          "distorsion_2x2.spline")
    imgm = np.mean([fabio.open(f).data for f in D_IR], 0).astype(np.float64)
    res = solve_image(imgm, (px1 + px2) / 2, lam=lam, phases=[cons],
                      solve_kw=dict(cascade=True, auto_center=False,
                                    refine_2d=False))
    assert res, "平均帧圆解失败"
    canvas, s = res["canvas"], res["s"]
    px_m = (px1 + px2) / 2 * s
    sol = res["sol"]
    cx, cy, D, m, rms = sol[:5]
    print(f"\n平均帧圆解: D={D*px_m*1000:.2f}mm 环数={m} rms={rms:.2f}px")
    fwd = make_pyfai_fwd(px1, px2, spline)
    tf = fit_tilt(canvas, cx, cy, D, Q, lam, px1 * s, px2 * s, sfull=s,
                  fwd=fwd, nsec=36, mass_rel=0.10, bump=1.5, dbg=True)
    if tf:
        out = write_poni(
            os.path.join(ROOT, "probes", "results", "alumina_mean.poni"),
            tf["dist"], tf["poni1"], tf["poni2"], tf["rot1"], tf["rot2"],
            lam, px1, px2, spline=spline,
            note="alumina 51-frame mean, B-route circle+tilt+spline")
        print(f"倾角: rot=({tf['rot1']*1e3:+.2f},{tf['rot2']*1e3:+.2f})mrad "
              f"(GT {rot1*1e3:+.2f},{rot2*1e3:+.2f}) "
              f"Δ=({(tf['rot1']-rot1)*1e3:+.2f},{(tf['rot2']-rot2)*1e3:+.2f})")
        print(f"D={tf['dist']*1000:.2f}mm (GT {gt*1000:.2f}, "
              f"Δ{(tf['dist']-gt)*1000:+.2f})  "
              f"poniΔ=({(tf['poni1']-float(txt['Poni1']))/px1:+.1f},"
              f"{(tf['poni2']-float(txt['Poni2']))/px2:+.1f})px  → {out}")

# ----------------------------------------------------------------- lab6
elif MODE == "lab6":
    # ⚠ 2026-10-31 结论：D2AM goniometer 数据集束流中心在探测器外 ~1.9m
    # （pyFAI 精化 poni1≈1950mm；CP 环是从底边扫过的弧，非同心环）。
    # 本工具（及 NN 训练域）假设束心在探测器内 → 密切圆伪解 D≈138mm，
    # 无几何意义。此数据集不适用，仅供 pyFAI API 演练，不进验收。
    import glob

    import pyFAI
    from pyFAI.control_points import ControlPoints
    from pyFAI.detectors import NexusDetector
    from pyFAI.geometryRefinement import GeometryRefinement

    D_IR = sorted(glob.glob(r"data\real\LaB6_gonio_D2AM\*.edf"))
    det = NexusDetector(r"data\real\LaB6_gonio_D2AM\D5Geom-2018.h5")
    print(f"Xpad: {det.shape} px={det.pixel1*1e6:.0f}µm")
    print(f"{'帧':>28}{'相':>7}{'D_mm':>8}{'Δref':>8}{'环数':>5}{'rms':>7}"
          f"{'ref环数':>7}")
    es = []
    for f in D_IR:
        npt = f[:-4] + ".npt"
        img = fabio.open(f).data.astype(np.float64)
        hdr = fabio.open(f).header
        lam = float(hdr["Lambda"]) * 1e-10
        # R3 统一入口：锁相 LaB6（solve_kw=旧 solve_auto 直调默认三关）
        res = solve_image(img, det.pixel1, lam=lam, phases=["LaB6"],
                          solve_kw=dict(cascade=False, auto_center=False,
                                        refine_2d=False), dbg=DBG)
        ph, sol = (res["phase"], res["sol"]) if res else (None, None)
        canvas, s = (res["canvas"], res["s"]) if res else (None, None)
        # 参考解：pyFAI 精化手工控制点（λ 固定）
        cp = ControlPoints(npt)
        data = cp.getList()          # [y(row), x(col), ring]
        n_ref_rings = len({d[2] for d in data})
        gr = GeometryRefinement(data, detector=det,
                                calibrant=cp.calibrant,
                                wavelength=lam, dist=0.5)
        try:
            gr.fixed = ["wavelength"]
        except Exception:
            pass
        gr.refine3(2000)
        ref_D = gr.dist
        name = os.path.basename(f)[:20]
        if sol is None:
            print(f"{name:>28}{'失败':>7}{'':>8}{ref_D*1000:8.1f}")
            es.append(999.0)
            continue
        cx, cy, D, m, rms = sol[:5]
        px_m = det.pixel1 * s
        e = D * px_m - ref_D
        es.append(abs(e) * 1000)
        print(f"{name:>28}{ph:>7}{D*px_m*1000:8.1f}{e*1000:+8.1f}"
              f"{m:>5}{rms:7.2f}{n_ref_rings:>7}")
    ok = [x for x in es if x < 900]
    print(f"\nLaB6 vs pyFAI参考: 平均|Δ|={np.mean(ok):.1f}mm "
          f"最差{max(ok):.1f}mm 成功{len(ok)}/{len(es)}")


