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

from solver_core import solve, solve_auto, to_canvas
from torchfai.phases import phase_qs

MODE = sys.argv[1] if len(sys.argv) > 1 else "ceria"
DBG = "dbg" in sys.argv[2:]


def acc(fit_mm, gt_mm):
    return fit_mm - gt_mm


# ---------------------------------------------------------------- ceria
if MODE == "ceria":
    import pyFAI
    import torch

    from real_refine_v6 import load_px_nn, to_nn_input_wide
    from torchfai.dataset import TorchFAIDataset as _T
    from torchfai.dataset import norm_to_param

    LAM = 0.972386
    PXEFF = 172e-6 / (521.0 / 2527)
    # 预注册同款 8 强环（phases.py 多的第 9 弱环 (422) I=8 会污染投票：
    # 其票散进中半径伪峰，挤出真簇——2026-10-31 回归教训）
    Q = phase_qs("CeO2")[:8]
    ds = _T(os.path.join(ROOT, "data", "train_v4w3"), "val")
    consts = ds.norm_consts
    m1, dev, _ = load_px_nn(r"runs\v4s1\best.pt"); m1.eval()

    print(f"{'帧':>5}{'fitΔmm':>9}{'环数':>5}{'rms_px':>8}")
    es = []
    ROT = []
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(
            ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = p.dist * 1000
        wide, k, extra_col_m = to_nn_input_wide(img)
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
        from solver_core import ring_radii
        peaks = ring_radii(wide, cx0, cy0)[0]
        sol = solve(peaks, cx0, cy0, D_px0, Q, LAM, PXEFF * 521.0 / Hm,
                    dbg=DBG, D_ref_m=p6[0], d_range=(0.35, 2.8),
                    rms_floor=1.8)
        if sol is None:
            print(f"{d:>5}  失败")
            es.append(999.0)
            continue
        cx, cy, D, m, rms = sol[:5]
        e = D * PXEFF * 521.0 / Hm * 1000 - gt
        es.append(abs(e))
        # ---- 倾角精化 + poni 输出（B 路线 Phase3：圆解→扇区控制点→LSQ）
        from poni_out import fit_tilt, write_poni
        pxm = PXEFF * 521.0 / Hm
        tf = fit_tilt(wide, cx, cy, D, Q, LAM, pxm, pxm, sfull=1,
                      nsec=36, mass_rel=0.10, bump=1.5, dbg=DBG)
        gr1 = float(p.rot1); gr2 = float(p.rot2)
        if tf:
            out = write_poni(
                os.path.join(ROOT, "probes", "results",
                             f"ceria_{d}.poni"),
                tf["dist"], tf["poni1"], tf["poni2"] + extra_col_m,
                tf["rot1"], tf["rot2"], LAM * 1e-10, 172e-6, 172e-6,
                note=f"ceria {d}mm, B-route circle+tilt")
            print(f"{d:>5}{e:+9.1f}{m:>5}{rms:8.2f}"
                  f"  rotΔ=({(tf['rot1']-gr1)*1e3:+5.1f},"
                  f"{(tf['rot2']-gr2)*1e3:+5.1f})mrad"
                  f" {tf['npts']}pt/{tf['nrings']}环"
                  f" rms3→5={tf['rms3']*1e3:.1f}→{tf['rms5']*1e3:.1f}"
                  f" DΔ={tf['dist']*1000-gt:+5.1f} → {out}")
            ROT.append(((tf['rot1']-gr1)*1e3, (tf['rot2']-gr2)*1e3))
        else:
            print(f"{d:>5}{e:+9.1f}{m:>5}{rms:8.2f}  倾角拟合失败(输出圆解)")
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
        canvas, s, cx0, cy0 = to_canvas(img)
        px_m = (px1 + px2) / 2 * s
        res = solve_auto(canvas, lam, px_m)
        picks.append(res[0] if res else None)
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
        canvas, s, cx0, cy0 = to_canvas(img)
        px_m = (px1 + px2) / 2 * s
        D_px0 = 1.0 / px_m
        from solver_core import ring_radii, solve
        peaks, prof, bin_px, good = ring_radii(canvas, cx0, cy0)
        sol = solve(peaks, cx0, cy0, D_px0, Q, lam, px_m,
                    I_rel=IR_rel, prof_ctx=(prof, bin_px, good),
                    canvas=canvas)
        name = os.path.basename(f)
        if sol is None or sol[3] < 5:
            print(f"{name:>18}{'失败':>8}")
            es.append(999.0)
            continue
        cx, cy, D, m, rms = sol[:5]
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
    canvas, s, cx0, cy0 = to_canvas(imgm)
    px_m = (px1 + px2) / 2 * s
    peaks, prof, bin_px, good = ring_radii(canvas, cx0, cy0)
    sol = solve(peaks, cx0, cy0, 1.0 / px_m, Q, lam, px_m, I_rel=IR_rel,
                prof_ctx=(prof, bin_px, good), canvas=canvas)
    assert sol, "平均帧圆解失败"
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
        canvas, s, cx0, cy0 = to_canvas(img)
        px_m = det.pixel1 * s
        res = solve_auto(canvas, lam, px_m, phases=["LaB6"], dbg=DBG)
        ph, sol = res if res else (None, None)
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
        e = D * px_m - ref_D
        es.append(abs(e) * 1000)
        print(f"{name:>28}{ph:>7}{D*px_m*1000:8.1f}{e*1000:+8.1f}"
              f"{m:>5}{rms:7.2f}{n_ref_rings:>7}")
    ok = [x for x in es if x < 900]
    print(f"\nLaB6 vs pyFAI参考: 平均|Δ|={np.mean(ok):.1f}mm "
          f"最差{max(ok):.1f}mm 成功{len(ok)}/{len(es)}")
