# -*- coding: utf-8 -*-
"""zoo_run.py：跨仪器零样本基准（路径2，15号协议）。

两步走（每套图）：
1) GT 自检（poni 锚定，独立于本组求解器）：pyFAI sload(poni) → integrate1d
   → find_peaks → 峰 q 与 6 相库逐一匹配（双向最近邻，1.5% 容差）→
   n_match≥4 且 中位相对误差<0.6% 判 PASS（= GT poni 与图像自洽 + 鉴定相）。
2) 零样本基准（仅 PASS 项）：to_canvas → solve_auto(cascade=True)（相盲、
   无初值、classic-first 生产路径）→ 报 ΔD、Δcenter、相鉴定、耗时、seed。

用法：python zoo_run.py            # 全部
      python zoo_run.py Pilatus1M  # 单项
输出：docs/zoo_benchmark_v0.md + docs/zoo_benchmark_v0.json
"""
import os
import sys
import time
import json
import traceback

import numpy as np
from scipy.signal import find_peaks

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)  # poni 内 spline 相对路径在此解析

import fabio
import pyFAI.azimuthalIntegrator
from torchfai.phases import PHASE_NAMES, phase_qs
from solver_core import (solve_image, solve_auto_lambda, to_canvas,
                         refine_lambda_d)

ITEMS = [  # (图, [poni...])
    ("Pilatus1M.edf", ["Pilatus1M.poni"]),
    ("Pilatus2M.cbf", ["Pilatus2M.poni"]),
    ("Pilatus6M.cbf", ["Pilatus6M.poni"]),
    ("Eiger4M.edf", ["Eiger4M.poni", "Eiger4M_v2.poni"]),
    ("Eiger9M.h5", ["Eiger9M.poni"]),
    ("Frelon2k.edf", ["Frelon2k.poni"]),
    # T2 去重（2026-11-01 深夜）：LaB6_0020.edf ≡ halfccd.edf（逐字节同一
    # 张图，poni 各异），保留 halfccd 一行，口径见表头。
    ("halfccd.edf", ["halfccd.poni"]),
    ("LaB6_260210.mar3450", ["Mar3450.poni"]),
    ("Fairchild.edf", ["Fairchild.poni"]),
]


def load_img(path):
    if path.endswith(".h5"):
        import h5py
        with h5py.File(path, "r") as f:
            # 通用探测：找首个名为 data 的 ≥2D 数据集（兼容
            # entry/data/data_000001 与 entry_0000/instrument/eiger/data）
            hit = []

            def _walk(name, obj):
                if not hit and isinstance(obj, h5py.Dataset) \
                        and obj.ndim >= 2 and name.split("/")[-1] == "data":
                    hit.append(obj)

            f.visititems(_walk)
            if not hit:
                raise IOError("no 2D 'data' dataset in h5")
            arr = hit[0][()]
            img = np.asarray(arr[0] if arr.ndim == 3 else arr, dtype=np.float64)
    else:
        img = np.asarray(fabio.open(path).data, dtype=np.float64)
    # 统一预处理（全 zoo 一致，文档记录）：哨兵/饱和值（≥1e9，Eiger4M 实测
    # 2^32 溢出像素 ~2300 个）与负 gap（Pilatus -2）置零。
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    return img


def gt_selfcheck(img, ai, wide=False):
    """poni 锚定相鉴定 + GT 自洽性。返回 (phase, n_match, med_rel, ok)。

    允许全局尺度因子 s（λ–D 简并：测试 poni 常用标称 λ 精化，D 吸收
    波长差 → 全部 q 同比偏移；Pilatus1M=AgBh 实测 s=1.032）。
    判据：≥3 环尺度后残差<1.2%，中位残差<0.8%，s∈[0.95,1.05]。
    """
    q_axis, I = ai.integrate1d(img, 4096, unit="q_A^-1")
    I = np.nan_to_num(I.astype(np.float64))
    if I.max() <= 0:
        return None, 0, 9.9, False, 1.0
    from scipy.ndimage import median_filter
    bg = median_filter(I, 121)
    d = I - bg
    mask = q_axis > 0.05
    if d[mask].max() <= 0:
        return None, 0, 9.9, False, 1.0
    pk, _ = find_peaks(d, prominence=0.04 * d[mask].max(), distance=6)
    pk = pk[mask[pk]]
    qpk = q_axis[pk]
    if len(qpk) < 3:
        return None, 0, 9.9, False, 1.0
    if wide:
        # 尺度扫描（λ 未知）：库线可整体落在量程外（LaB6_0020 量程
        # qmax=1.86 < LaB6 首线 2.014，真 λ=0.144Å）——逐线最近峰中位
        # 比失效，改为 s 网格上数匹配。score=n 主、中位残差次。
        best = (None, -99, 9.9, 1.0)  # (ph, cov, med, s)
        for ph in PHASE_NAMES:
            qs = np.asarray(phase_qs(ph), float)
            if len(qs) < 3:
                continue
            for s in np.geomspace(0.10, 4.5, 90):
                pred = s * qs
                rel = np.abs(qpk[:, None] / pred[None, :] - 1.0)
                near = rel.min(axis=1)
                hitmask = near < 0.012
                n = int(hitmask.sum())
                if n < 3:
                    continue
                vis = int((pred < q_axis.max() * 1.05).sum())
                cov = n - 0.5 * max(vis - n, 0)
                med = float(np.median(near[hitmask]))
                if cov > best[1] or (cov == best[1] and med < best[2]):
                    best = (ph, cov, med, float(s))
        ph, cov, med, s0 = best
        n = max(int(cov + 0.5 * 1), 0)  # 报告用近似环数
        ok = (ph is not None and med < 0.008 and 0.10 < s0 < 4.5)
        return ph, n, med, ok, s0
    best = (None, 0, 9.9, 1.0)
    for ph in PHASE_NAMES:
        qs = np.asarray(phase_qs(ph), float)
        qs = qs[qs < q_axis.max() * 1.02]
        if len(qs) < 3:
            continue
        # 粗尺度：每个库环最近峰的比值取中位
        ratios = []
        for qa in qs:
            i = np.argmin(np.abs(qpk - qa))
            if abs(qpk[i] - qa) / qa < 0.10:
                ratios.append(qpk[i] / qa)
        if len(ratios) < 3:
            continue
        s0 = float(np.median(ratios))
        rel = np.abs(qpk[:, None] / (s0 * qs[None, :]) - 1.0)
        near = rel.min(axis=1)
        n = int((near < 0.012).sum())
        med = float(np.median(near[near < 0.012])) if n else 9.9
        if n > best[1] or (n == best[1] and med < best[2]):
            best = (ph, n, med, s0)
    ph, n, med, s0 = best
    if wide:
        ok = (n >= 3 and med < 0.008 and 0.10 < s0 < 4.5)
    else:
        ok = (n >= 3 and med < 0.008 and 0.95 < s0 < 1.05)
    return ph, n, med, ok, s0



def _legacy_canvas_fallback_removed():
    """2026-11-05 R1/R2/R3（18-1104 需求）：canvas_of(723) 与
    _fallback_joint(ceil(maxdim/2)) 已删除——画布策略与 fallback 门控
    统一收敛到 solver_core.canvas_of / solve_image（默认档 520 +
    零样本安全门 a/b/c）。历史口径见 zoo_benchmark_v3.md 的 T4 注与
    docs/unify_canvas_fallback.md。"""
    return None


def main(sel=None, lam_map=None):
    """lam_map（T8 辅助模式）：{tag前缀: λ(Å)}——只跑映射命中的行，
    求解用用户 λ（外部信息硬破 λ–D 简并），产出
    docs/zoo_benchmark_assist.{md,json}；零样本 v3 路径一字不变。"""
    rows = []
    out_json = []
    for img_f, ponis in ITEMS:
        if sel and sel not in img_f:
            continue
        try:
            img = load_img(os.path.join(ZOO, img_f))
        except Exception as e:
            rows.append(f"| {img_f} | LOAD FAIL: {e} | | | | |")
            continue
        for poni_f in ponis:
            tag = f"{img_f.split('.')[0]}" + (
                "_v2" if "v2" in poni_f else "")
            lam_eff = None                     # T8：用户 λ（辅助模式）
            if lam_map is not None:
                for _k, _v in lam_map.items():
                    if tag == _k or tag.startswith(_k):
                        lam_eff = float(_v)
                        break
                if lam_eff is None:
                    continue                   # 辅助模式只跑映射行
            try:
                ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
                if not ai.wavelength or ai.wavelength <= 0:
                    # ---- 无 λ 分支：联合 λ–D 精化（W9 v1）----
                    # ①自检：λ 置 1Å，尺度窗放宽到 [0.25,4]（s 吸收 λ 比，
                    #   q 比指纹相鉴不变）；②联合求解 solve_auto_lambda；
                    # ③验证=像素空间几何一致性：同用 λ_fit 比较我们的
                    #   (D_fit,center) 与 GT poni 的 (D_GT,center_GT) 对
                    #   相线系的预测半径（零倾 GT 精确，小角简并族天然等价）
                    ai.wavelength = 1e-10
                    ph, n, med, ok, s0 = gt_selfcheck(img, ai, wide=True)
                    if not ok:
                        rows.append(
                            f"| {tag} | ✗ 自检失败({ph},{n}环,{med*100:.2f}%)"
                            f" | — | — | — | — | — | — | — | — |")
                        out_json.append(dict(tag=tag, ok=False,
                                             reason="selfcheck-no-lam"))
                        continue
                    # ---- 无 λ 分支求解：solve_image 统一入口（联合 λ–D
                    # + R1/R2 画布与 fallback 策略；Frelon2k/halfccd 的
                    # s=4 两档同 s=2 差异、(b) 门不触发 → 行为与旧直调
                    # solve_auto_lambda 一致，2026-11-05 校准实证）----
                    px_native = float(ai.detector.pixel1)
                    t0 = time.perf_counter()
                    res = solve_image(img, px_native, lam=None)
                    dt = (time.perf_counter() - t0) * 1000
                    if res is None:
                        rows.append(f"| {tag} | ✔ {ph}({n}环) | "
                                    f"JOINTλ SOLVE FAIL | — | — | — | "
                                    f"— | — | — | — | — |")
                        out_json.append(dict(tag=tag, ok=False,
                                             reason="joint-solve-fail"))
                        continue
                    canvas, s, px_m = (res["canvas"], res["s"],
                                       res["px_m"])
                    ph_fit, sol, lam_fit = (res["phase"], res["sol"],
                                            res["lam"])
                    cx, cy, D_px, m, rms = sol[0], sol[1], sol[2], sol[3], sol[4]
                    D_fit = D_px * px_m * 1000.0
                    # GT 侧几何（λ=λ_fit）：FT2D 平几何 r = D·tan(2θ)/px
                    try:
                        f2d = ai.getFit2D()
                        gt_cx, gt_cy = f2d["centerX"], f2d["centerY"]
                        gt_D_mm = ai.dist * 1000.0
                    except Exception:
                        gt_cx = ai.poni2 / ai.detector.pixel2
                        gt_cy = ai.poni1 / ai.detector.pixel1
                        gt_D_mm = ai.dist * 1000.0
                    qs = np.asarray(phase_qs(ph_fit), float)
                    tt = 2.0 * np.arcsin(
                        np.clip(lam_fit * qs / (4.0 * np.pi), 0, 0.9999))
                    r_our = (D_fit / 1000.0) * np.tan(tt) / px_m   # 画布 px
                    r_gt = (gt_D_mm / 1000.0) * np.tan(tt) / px_m  # 同λ同px
                    vis = (r_gt > 25) & (r_gt < 0.5 * max(canvas.shape))
                    dr = np.abs(r_our[vis] - r_gt[vis]) if vis.any() else np.array([9.9])
                    dcx = cx - gt_cx / s
                    dcy = cy - gt_cy / s
                    dcp = float(np.hypot(dcx, dcy))
                    # T9：无 λ 分支同口径——Δc 对共心真值，GTΔc 归因列
                    dc_cell, gt_dc_cell, cref_j = f"{dcp:.0f}ᴳᵀ", "—", {}
                    try:
                        from solver_core import concentric_center as _cc
                        _cr = _cc(img, (cx * s, cy * s), D_px * s, qs,
                                  lam_fit)
                    except Exception:
                        _cr = None
                    if _cr is not None:
                        _dc9 = float(np.hypot(cx * s - _cr["cx"],
                                              cy * s - _cr["cy"]))
                        _gd9 = float(np.hypot(gt_cx - _cr["cx"],
                                              gt_cy - _cr["cy"]))
                        dc_cell = f"{_dc9:.1f}"
                        gt_dc_cell = f"{_gd9:.0f}"
                        cref_j = dict(
                            center_ref=[_cr["cx"], _cr["cy"]],
                            cref_sigma=[_cr["sx"], _cr["sy"]],
                            cref_n=int(_cr["n"]),
                            dcenter_ref_px=_dc9,
                            gt_dcenter_ref_px=_gd9)
                    geom_ok = bool(np.median(dr) < 1.0 and dr.max() < 2.5
                                   and dcp < 8.0)
                    hit = "✔" if ph_fit == ph else "✗"
                    mark = "✔" if (geom_ok and ph_fit == ph) else (
                        "◐" if geom_ok else "✗")
                    # 域界归因：三套无λ图全带样条畸变（halfccd.spline 等），
                    # 平几何模型的径向谱被畸变撕裂成扇区峰（LaB6_0020 实测
                    # geo 处峰表与 GT 处完全不同），索引错位族横行——层-1
                    # 畸变估计是开放问题，不是 λ 问题。
                    rows.append(
                        f"| {tag} | ✔ {ph}({n}环) | {ph_fit}{hit} "
                        f"m={m} {mark}"
                        f"(一致性中位{np.median(dr):.1f}/"
                        f"最大{dr.max():.1f}px;样条畸变域界) | "
                        f"{gt_D_mm:.1f}(D@λ̂) | — | {dc_cell} | "
                        f"{gt_dc_cell} | {lam_fit:.4f} | "
                        f"{D_fit - gt_D_mm:+.2f} | s={s} |")
                    out_json.append(dict(
                        tag=tag, ok=True, no_lambda=True, phase=str(ph),
                        tier=f"s={s}",
                        n=n, lam_fit=float(lam_fit), phase_fit=str(ph_fit),
                        D_fit=float(D_fit), gt_D_at_lamfit=float(gt_D_mm),
                        dD_lamhat=float(D_fit - gt_D_mm),
                        med_dr=float(np.median(dr)), max_dr=float(dr.max()),
                        dcenter_px=dcp, geom_ok=geom_ok, ms=float(dt),
                        **cref_j))
                    continue
                gt_D_mm = ai.dist * 1000.0
                try:
                    f2d = ai.getFit2D()
                    gt_cx, gt_cy = f2d["centerX"], f2d["centerY"]
                except Exception:
                    gt_cx = ai.poni2 / ai.detector.pixel2
                    gt_cy = ai.poni1 / ai.detector.pixel1
                ph, n, med, ok, s0 = gt_selfcheck(img, ai)
                if not ok:
                    # T2：Pilatus6M 为非标定帧（GT 自检即不可鉴），不进
                    # 成功率分母——域界归因，与求解器无关。
                    p6m = "；非标定帧(不进分母)" if "Pilatus6M" in tag else ""
                    rows.append(
                        f"| {tag} | ✗ 自检失败({ph},{n}环,{med*100:.2f}%)"
                        f"{p6m} | — | — | — | — | — | — | — | — | — |")
                    out_json.append(dict(tag=tag, ok=False, phase=str(ph),
                                         n=n, med_rel=med, scale=s0,
                                         excluded_noncalib="Pilatus6M" in tag))
                    continue
                # ---- 零样本基准：solve_image 统一入口（R1 画布默认档
                # 520 + R2 零样本安全 fallback；旧 T4 的"相错 AND ps<
                # 0.55"门依赖 GT，已废）----
                px_native = float(ai.detector.pixel1)
                lam_A = ai.wavelength * 1e10
                t0 = time.perf_counter()
                res = solve_image(
                    img, px_native, lam=(lam_eff if lam_eff else lam_A),
                    fb_grid=((0.96, 1.04, 9) if lam_map is not None
                             else None))          # T8 辅助=锁相网格
                dt = (time.perf_counter() - t0) * 1000
                if res is None:
                    rows.append(f"| {tag} | ✔ {ph}({n}环) | SOLVE FAIL"
                                f" | {gt_D_mm:.1f} | — | — | — | — | — |"
                                f" — | — |")
                    out_json.append(dict(tag=tag, ok=True, phase=str(ph),
                                         gt_D=gt_D_mm, solved=False))
                    continue
                canvas, s, px_m = res["canvas"], res["s"], res["px_m"]
                ph_fit, sol = res["phase"], list(res["sol"])
                use_fb = res["used_fallback"]
                lam_fb = res["lam"] if use_fb else None   # 网格 λ̂（简并脊）
                tier = f"s={s}"
                if use_fb:
                    tier = (f"s={s}·λu锁相" if lam_map is not None
                            else f"s={s}·λ̂")
                    if lam_map is not None:
                        # T8：锁相在 λ_fit 完成，D 按积守恒换算回 λu
                        # （小角 r∝λD；环半径一致，D@λu = D_fit·λ_fit/λu）
                        sol[2] = sol[2] * (lam_fb / lam_eff)
                lam_lock = float(lam_fb) if (use_fb and lam_map is not None
                                             and lam_fb is not None) else None
                lam_used = (lam_eff if lam_map is not None
                            else (lam_fb if use_fb else lam_A))
                cx, cy, D_px, m, rms, icorr, pscore = sol[:7]
                D_fit = D_px * px_m * 1000.0
                dD = D_fit - gt_D_mm
                dcx = cx * s - gt_cx
                dcy = cy * s - gt_cy
                dcp = float(np.hypot(dcx, dcy))
                hit = "✔" if ph_fit == ph else "✗"
                # T3：2D 逐环 Kasa 精化痕迹（solve_auto 在 ps>0.4 且 m≥4
                # 时自动调用；sol 尾部追加 pre_cx, pre_cy, pre_D_px, n_rings）
                r2d_n = int(sol[10]) if len(sol) >= 11 else 0
                r2d_pre_dc = None
                if r2d_n:
                    tier += "·2D"
                    r2d_pre_dc = float(np.hypot(sol[7] * s - gt_cx,
                                                sol[8] * s - gt_cy))
                # T7（2026-11-01 深夜）：原生分辨率二次精化——画布 2D 解
                # 为初始，原生图 refine_geometry_2d（half=3 原生px），成功
                # 则写回 sol，下游（倾角/角域/λ̂/行值）全链一致。探针
                # _t7_native.py v2：P2M dc 2.99→1.12px、E9M 9.25→6.19px
                # dD +0.03→+0.01、mar 8.32→6.65px（脊上 +6.79→+5.87）；
                # Eiger4M/v2 只罩 2 条完整环→诚实拒收（拒收=行不变）。
                # v1 教训：必须用本文件 load_img 的原生图（E9M uint32
                # 哨兵像素 4294967295 若清零，解掉错盆地 dc 92.9 vs 9.25）。
                r2d_nat, dcp_nat, dD_nat = 0, None, None
                if r2d_n:
                    try:
                        from solver_core import refine_geometry_2d as _rg2
                        _q7 = np.asarray(phase_qs(ph_fit), float)
                        _o = _rg2(img, sol[0] * s, sol[1] * s, sol[2] * s,
                                  _q7, lam_used, n_iter=3)
                    except Exception:
                        _o = None
                    if _o is not None:
                        _cx2, _cy2, _D2, _nr = _o
                        r2d_nat = int(_nr)
                        dcp_nat = float(np.hypot(_cx2 - gt_cx,
                                                 _cy2 - gt_cy))
                        dD_nat = _D2 * (px_m / s) * 1000.0 - gt_D_mm
                        sol = list(sol)
                        sol[0], sol[1], sol[2] = _cx2 / s, _cy2 / s, _D2 / s
                        cx, cy, D_px = sol[0], sol[1], sol[2]
                        D_fit = D_px * px_m * 1000.0
                        dD = D_fit - gt_D_mm
                        dcx = cx * s - gt_cx
                        dcy = cy * s - gt_cy
                        dcp = float(np.hypot(dcx, dcy))
                        tier += "ⁿ"
                # ---- T9 共心中心真值（2026-11-03）：Δcenter 改对照不依赖
                # GT 的 center_ref（所有环共用圆心，径向散布最小点）；
                # GT 中心差单独成列（GTΔc）暴露 GT 污染幅度。ceria 判决：
                # GT 中心系统偏 15.5–17.9px（σ_ref 0.02px、GT 中心积分
                # 环峰高仅赢者 1/3~1/5）——旧 Δcenter 列在 GT 差的仪器上
                # 读数被污染。共心拟合失败/环不足 → 退回 GT 口径并标 ᴳᵀ。
                dc_cell, gt_dc_cell, cref_j = f"{dcp:.0f}ᴳᵀ", "—", {}
                try:
                    from solver_core import concentric_center as _cc
                    _q9 = np.asarray(phase_qs(ph_fit), float)
                    _cr = _cc(img, (cx * s, cy * s), D_px * s, _q9,
                              lam_used)
                except Exception:
                    _cr = None
                if _cr is not None:
                    _dc9 = float(np.hypot(cx * s - _cr["cx"],
                                          cy * s - _cr["cy"]))
                    _gd9 = float(np.hypot(gt_cx - _cr["cx"],
                                          gt_cy - _cr["cy"]))
                    dc_cell = f"{_dc9:.1f}"
                    gt_dc_cell = f"{_gd9:.0f}"
                    cref_j = dict(center_ref=[_cr["cx"], _cr["cy"]],
                                  cref_sigma=[_cr["sx"], _cr["sy"]],
                                  cref_n=int(_cr["n"]),
                                  dcenter_ref_px=_dc9,
                                  gt_dcenter_ref_px=_gd9)
                # 倾角验证（W9 四役）：胜项跑控制点倾角拟合 vs GT rot1/2
                # E1f3d（2026-11-03 深夜）：原生图拟合优先、画布兜底——
                # 池化画布会压低 imp 信号且大图上可能整体失败（E9M canvas
                # tf=None 列空既有问题；native imp=34%、Δc(tilt)=0.66px，
                # _nc_e1f3.py）。px=px_m/s（画布像素含池化因子 s）。
                tilt_s = ""
                try:
                    from poni_out import fit_tilt as _ft
                    qs_w = np.asarray(phase_qs(ph_fit), float)
                    try:
                        tf = _ft(img, cx * s, cy * s, D_px * s, qs_w,
                                 lam_used, px_m / s, px_m / s, sfull=1,
                                 nsec=36, mass_rel=0.10, bump=1.5)
                    except Exception:
                        tf = None
                    if tf is None:
                        tf = _ft(canvas, cx, cy, D_px, qs_w, lam_used,
                                 px_m, px_m, sfull=s,
                                 nsec=36, mass_rel=0.10, bump=1.5)
                    if tf:
                        dr1 = (tf["rot1"] - ai.rot1) * 1000
                        dr2 = (tf["rot2"] - ai.rot2) * 1000
                        tilt_s = f" Δrot=({dr1:+.1f},{dr2:+.1f})mrad"
                        # ---- E1f3c tilt 中心列（2026-11-03 深夜）：V3 成对
                        # 几何（D=解算 + tf poni/rot）的等效 Fit2D 中心 vs
                        # GT-Fit2D 中心（锐度最优口径；_nc_e1f3*.py 判决：
                        # E9M 0.66px / ceria_400 2.5px / Eiger4M 9.3px，
                        # 替代被共心 ref 偏置污染的旧中心列读数）。
                        try:
                            from pyFAI.integrator.azimuthal import \
                                AzimuthalIntegrator as _AI
                            from pyFAI.detectors import Detector as _Det
                            _pxn = px_m / s
                            _ai3 = _AI(dist=D_px * px_m,
                                       poni1=tf["poni1"],
                                       poni2=tf["poni2"],
                                       rot1=tf["rot1"], rot2=tf["rot2"],
                                       rot3=0.0,
                                       wavelength=lam_used * 1e-10,
                                       detector=_Det(pixel1=_pxn,
                                                     pixel2=_pxn))
                            _f3 = _ai3.getFit2D()
                            _gf = ai.getFit2D()
                            _dctr = float(np.hypot(
                                _f3["centerX"] - _gf["centerX"],
                                _f3["centerY"] - _gf["centerY"]))
                            tilt_s += f" Δc(tilt)={_dctr:.1f}px"
                        except Exception:
                            pass
                except Exception:
                    pass
                # ---- T6 判决实验（2026-11-01 深夜）：λ 可辨角域律 ----
                # 控制点联合 (λ,D,poni,rot1,rot2) 拟合实证：λ̂ σ<2% ⟺
                # 环系实际 2θmax≳30°（Eiger4M 36°→σ1.0%）；角域窄
                # （mar3450 17.6°、Eiger9M 20°）时固定λ-rms曲线无内部
                # 极小、λ 数学上不可辨（简并非噪声所致）。λ̂ 列因此追加
                # "角域X°"——把 σ% 神秘数字变成自解释域界。探针
                # _t_tilt_joint{,2}.py。
                ttm = None
                try:
                    from poni_out import sector_peaks as _sp
                    qs_w2 = np.asarray(phase_qs(ph_fit), float)
                    tt_w = 2 * np.arcsin(np.clip(
                        lam_used * qs_w2 / (4.0 * np.pi), 0, 0.9999))
                    rad_w = D_px * np.tan(tt_w)
                    rad_w = rad_w[(rad_w > 25) & (rad_w < 0.98 * min(
                        canvas.shape))]
                    _pts = _sp(canvas, cx, cy, rad_w, nsec=12, win=6.0,
                               mass_rel=0.10, bump=1.5)
                    if _pts:
                        rr_max = max(p[2] for p in _pts)
                        # r/D_px = tan(2θ) → 2θ = arctan(r/D)，单个 arctan
                        # （首版误写 2·arctan 全表虚高一倍，E4M 71°≠36°）
                        ttm = float(np.degrees(
                            np.arctan(rr_max / D_px)))
                except Exception:
                    ttm = None
                ttm_s = f"·角域{ttm:.0f}°" if ttm else ""
                # ---- λ̂ 报告（T1，2026-11-01 深夜）：有 λ 分支联合复核 ----
                # 名义 λ 有偏的图（Pilatus1M 自检 s=1.0326），名义 λ 下 ΔD
                # 被 λ–D 简并整体吞掉（+56mm）；窄窗 [0.90,1.10]×λ₀ 联合
                # λ–D（solve_auto_lambda 三段收口 + 连续 λ-D LSQ）给出
                # λ̂ 与 D@λ̂，与无 λ 三套同口径：D@λ̂ 对 GT 名义 D，
                # r_our/r_gt 像素一致性同用 λ̂。零样本：GT 只在评估侧。
                # 简并情形（σ(λ)/λ≥2%，环太少/角太窄，Pilatus1M 3 个小角
                # 谐波环实证 σ=4.8）：λ̂ 不可辨，报"简并"+简并积偏差
                # D_fit·λ₀ vs gt_D·(s0·λ₀)——这才是该图零样本的物理极限。
                lam_cell, dlam_cell, j2 = "—", "—", {}
                if lam_map is not None:
                    # ---- T8 辅助模式：λ 用户给定，不做 λ̂ 搜索 ----
                    lam_cell = (f"λu={lam_eff:.4f}"
                                + (f"·角域{ttm:.0f}°" if ttm else ""))
                    dlam_cell = "—(辅助)"
                    j2 = dict(lam_src="user", lam_hat=float(lam_eff),
                              tthmax_deg=ttm, lam_lock=lam_lock)
                elif use_fb:
                    # ---- T4 fallback λ̂ 报告：λ̂=联合网格格点（简并脊上，
                    # 位置由网格步长定，任意）；refine_lambda_d 尝试连续
                    # 收口，σ(λ)/λ≥2% 按门控拒收（mar3450 深夜实证
                    # σ≈44%）→ 标"简并"并回退网格 λ̂。
                    lr = None
                    try:
                        lr = refine_lambda_d(canvas, cx, cy, D_px, lam_fb,
                                             phase_qs(ph_fit),
                                             lam_frac=0.02)
                    except Exception:
                        lr = None
                    if lr is not None and lr[4] is not None and lr[4] < 0.02:
                        lam_hat = lr[0]
                        D_lam = lr[1] * px_m * 1000.0
                        sig = None
                    else:
                        lam_hat, D_lam = lam_fb, D_fit
                        sig = lr[4] if lr is not None else None
                    dD_lam = D_lam - gt_D_mm
                    qs2 = np.asarray(phase_qs(ph_fit), float)
                    tt = 2.0 * np.arcsin(np.clip(
                        lam_hat * qs2 / (4.0 * np.pi), 0, 0.9999))
                    r_our = (D_lam / 1000.0) * np.tan(tt) / px_m
                    r_gt = (gt_D_mm / 1000.0) * np.tan(tt) / px_m
                    vis = (r_gt > 25) & (r_gt < 0.5 * max(canvas.shape))
                    dr = (np.abs(r_our[vis] - r_gt[vis]) if vis.any()
                          else np.array([9.9]))
                    sig_s = f"·简并(σ{sig*100:.0f}%)" if sig else ""
                    lam_cell = f"{lam_hat:.4f}{sig_s}{ttm_s}"
                    dlam_cell = f"{dD_lam:+.2f}"
                    j2 = dict(lam_hat=float(lam_hat),
                              phase_lamhat=str(ph_fit),
                              tthmax_deg=ttm,
                              D_at_lamhat=float(D_lam),
                              dD_lamhat=float(dD_lam),
                              lam_dr_med=float(np.median(dr)),
                              lam_dr_max=float(dr.max()),
                              lam_sig=sig)
                else:
                    t1 = time.perf_counter()
                    r2 = solve_auto_lambda(
                        canvas, px_m, lam_lo=lam_A * 0.90, lam_hi=lam_A * 1.10,
                        n_s=5, auto_center=True,
                        refine_stages=((0.04, 7), (0.008, 7)), full=True)
                    ms2 = (time.perf_counter() - t1) * 1000
                    if r2 is not None:
                        lam_hat = r2["lam"]
                        if r2["lam_src"] == "lsq":
                            ph2, sol2 = r2["phase"], r2["sol"]
                            D_lam = sol2[2] * px_m * 1000.0
                            dD_lam = D_lam - gt_D_mm
                            qs2 = np.asarray(phase_qs(ph2), float)
                            tt = 2.0 * np.arcsin(np.clip(
                                lam_hat * qs2 / (4.0 * np.pi), 0, 0.9999))
                            r_our = (D_lam / 1000.0) * np.tan(tt) / px_m
                            r_gt = (gt_D_mm / 1000.0) * np.tan(tt) / px_m
                            vis = (r_gt > 25) & (r_gt < 0.5 * max(canvas.shape))
                            dr = (np.abs(r_our[vis] - r_gt[vis]) if vis.any()
                                  else np.array([9.9]))
                            lam_cell = f"{lam_hat:.4f}{ttm_s}"
                            dlam_cell = f"{dD_lam:+.2f}"
                            j2 = dict(lam_hat=float(lam_hat),
                                      phase_lamhat=str(r2["phase"]),
                                      tthmax_deg=ttm,
                                      D_at_lamhat=float(D_lam),
                                      dD_lamhat=float(dD_lam),
                                      lam_dr_med=float(np.median(dr)),
                                      lam_dr_max=float(dr.max()),
                                      ms_lamhat=float(ms2))
                        else:
                            prod_dev = D_fit / (gt_D_mm * s0) - 1.0
                            sig_s = (f"{r2['lam_sig']*100:.0f}%"
                                     if r2["lam_sig"] else "?")
                            ang_s = f",角域{ttm:.0f}°" if ttm else ""
                            lam_cell = f"简并(σ{sig_s}{ang_s})"
                            dlam_cell = f"积偏差{prod_dev:+.2%}"
                            j2 = dict(lam_hat=None, lam_src="degenerate",
                                      tthmax_deg=ttm,
                                      lam_sig=r2["lam_sig"],
                                      prod_dev=float(prod_dev),
                                      ms_lamhat=float(ms2))
                # T8 辅助模式：fallback 行 ΔD 也报（D@λu 是本模式的点），
                # 零样本 fallback 行才置 —。
                dD_cell = (f"{dD:+.2f}" if (lam_map is not None or not use_fb)
                           else "—")
                rows.append(
                    f"| {tag} | ✔ {ph}({n}环) | {ph_fit}{hit} "
                    f"m={m} rms={rms:.1f} | {gt_D_mm:.1f} | "
                    f"{dD_cell} | {dc_cell} | {gt_dc_cell} | {lam_cell} |"
                    f" {dlam_cell} |{tilt_s} | {tier} |")
                out_json.append(dict(
                    tag=tag, ok=True, phase=str(ph), n=n, med_rel=med,
                    scale=s0,
                    gt_D=gt_D_mm, solved=True, phase_fit=str(ph_fit),
                    m=int(m), rms=float(rms), dD=float(dD),
                    dcx=float(dcx), dcy=float(dcy), dcenter_px=dcp,
                    ms=float(dt), shape=list(img.shape), tier=tier,
                    fallback=bool(use_fb), fb_reason=str(res["fb_reason"]),
                    def_ps=res["def_ps"], def_m=res["def_m"],
                    def_rms=res["def_rms"], lrg_ps=res["lrg_ps"],
                    lrg_m=res["lrg_m"], lrg_rms=res["lrg_rms"],
                    r2d_n=int(r2d_n),
                    dcenter_pre_px=r2d_pre_dc,
                    r2d_native=int(r2d_nat),
                    dcenter_native_px=dcp_nat, dD_native=dD_nat,
                    gt_center=[float(gt_cx), float(gt_cy)], **cref_j, **j2))
            except Exception:
                rows.append(f"| {tag} | ERROR | ```{traceback.format_exc()[-300:]}"
                            f"``` | | | | | | | | | |")
    print("\n".join(rows))
    md = ["# 跨仪器零样本基准 v3（silx zoo, classic-first, 相盲无初值）",
          "", "生成：zoo_run.py ｜ GT=目录 poni（pyFAI sload, SI 单位）",
          "> 2026-11-01 晚复核：新增 auto_center（束在画布一角时梯度投票定心）；Pilatus1M 相鉴定已对（AgBh），+56mm 为 AgBh 谐波序列的 λ–D 绝对尺度退化（GT 自检 s=1.0326 同量级），非定心/提取失败。",
          "> 2026-11-01 深夜 T1：λ̂ 报告列。solve_auto_lambda 改三段收口（粗网格记 top-3 → ±frac₁ 7 点 → ±frac₂ 7 点收口）+ 连续 λ-D LSQ（refine_lambda_d，σ(λ)/λ 由 JᵀJ 反演，≥2% 判简并）；有 λ 分支加窄窗 [0.90,1.10]×λ₀ 联合复核——λ–D 简并下名义 ΔD 被 λ 偏差吞掉时（Pilatus1M s=1.033），λ̂/D@λ̂ 给出同口径几何，简并时报积偏差 D_fit·λ₀ vs gt_D·(s0·λ₀)（零样本物理极限）。λ̂ 与 D@λ̂ 全零样本，GT 只在评估侧；判定列（相/ΔD/Δcenter）仍以名义 λ 解为准。",
          "> 2026-11-01 深夜 T2（口径清理）：halfccd ≡ LaB6_0020（逐字节同一张图，poni 各异）去重，保留 halfccd 一行；Pilatus6M 标非标定帧、不进分母。**口径：可评 8 套**（过 GT 自检：Pilatus1M / Pilatus2M / Eiger4M / Eiger4M_v2 / Eiger9M / Frelon2k / halfccd / mar3450）；**域界 2**（Pilatus6M 非标定帧、Fairchild 束在画布外）；**4 胜** = 相对+亚毫米（Eiger4M / v2 / Eiger9M / Pilatus2M）。",
          "> 2026-11-01 深夜 T4（画布档位列）：默认档（max_side=520）**相错 AND ps<0.55** 的大图触发 fallback——s=⌈maxdim/2⌉ 大画布 + λ̂ 联合网格 geomspace(0.95,1.05,7)×λ₀，按 (ps,m,−rms) 取优；胜出行口径切 λ̂ 解（ΔD 置 —，几何进 λ̂/ΔD@λ̂ 列）。触发阈不裸用 ps<0.55：胜项 ps 实测 0.446–0.476，裸阈会误伤 P2M/Eiger4M/Eiger9M。mar3450 经此出 LaB6（λ̂ 为简并脊网格点，ΔD@λ̂ 如实报，不修尺度）。",
          "> 2026-11-01 深夜 T3（2D 逐环 Kasa 精化）：solve_auto 新增 refine_2d 开关（ps>0.4 且 m≥4 自动调用 refine_geometry_2d：相线系预测环带内亮像元 → 强度加权 Kasa 圆拟合（2.5σ 剪枝）→ 0.9·r_edge 边缘门只收完整环 → 环半径对 D 加权 LSQ）。zoo_run 全档开启（画布列 ·2D 标记）：Pilatus2M ΔD +1.10→−0.05、Eiger9M +0.19→+0.03；mar3450 fallback 解 12 环精化 Δcenter 11→8px、ΔD@λ̂ +8.46→+6.80。Eiger4M/v2 探测器仅罩 2 条完整环→门控拒收（行不变）；Pilatus1M m=3 低于门（行不变）。中心 <1px 目标未达（画布池化系统差），如实记。",
          "> 2026-11-01 深夜 T6（λ 可辨角域律——判决实验）：控制点（扇区峰位）联合 (λ,dist,poni,rot1,rot2) LSQ 判决——**λ 可辨 ⟺ 环系实际 2θmax≳30°**。阳性对照 Eiger4M（角域 36°）：固定λ-rms 曲线清晰内部极小 @0.9218，联合 λ̂=0.9279 σ=1.0%；阴性 mar3450（17.6°）/Eiger9M（20°，环只到 145 画布px）：曲线单调无极小、λ̂ 撞界。**简并非噪声或倾角未建模，而是角域不足**——λ̂ 列追加\"角域X°\"（扇区峰位支持的最大环半径 2θ），σ% 神秘数字变自解释域界。探针 _t_tilt_joint{,2}.py。",
          "> 2026-11-01 深夜 T7（原生分辨率二次精化）：画布 2D 解为初始（cx·s,cy·s,D·s），原生图 refine_geometry_2d（half=3 原生px）成功则写回 sol——下游倾角/角域/λ̂/行值全链一致，tier 加 ⁿ。战绩：Pilatus2M Δcenter 2.99→**1.12px**、Eiger9M 9.25→**6.19px**（ΔD +0.03→+0.01）、mar3450 8.32→**6.65px**（ΔD@λ̂ +6.80→+5.87）；Eiger4M/v2 只罩 2 条完整环→诚实拒收行不变。E9M 哨兵像素教训：原生图必须与 load_img 同口径（uint32 哨兵清零会掉错盆地）。探针 _t7_native.py。",
          "> 2026-11-02 T5（样条判决——负结果+归因反转）：solver_core.undistort 严格逆映射双验证（合成往返 band|Δ| med 0.05px；inv+平几何自检≡raw+spline 自检 0.66/0.3312 vs 0.69/0.3312）。**两行均非样条主罪**：halfccd 图字节≡LaB6_0020、平几何 LaB6@0.1899 med 0.22% 比 poni(spline) 的 Al2O3 0.69% 紧 3 倍 → GT 口径 artifact（图本质未畸变）；Frelon2k 环本征弱宽（21 宽包，畸变致宽≈本征宽度，校正后锐度 1.9<2.6 无增益）→ 提取极限（mar 动态范围同类）。零样本协议求解器不见 poni，spline 属 GT 侧。探针 _t5_{inv,roundtrip,sharp,half2,diag2}.py。",
          "> 2026-11-03 T9（共心中心真值）：Δcenter 改对照 **center_ref**（所有环共用圆心、径向散布最小点；solver_core.concentric_center，σ 由曲率估计）——不依赖 GT；新列 **GTΔc** = GT poni 中心对 center_ref 的偏差，暴露 GT 污染幅度。动机：ceria 判决实验（_center_truth/_center_gt.py）实证 GT 中心系统偏 15.5–17.9px（σ_ref 0.02–0.03px 三档互洽；GT 中心提谱首环峰高仅最优中心的 1/3~1/5、宽 18px vs 12–13px）。共心环不足（<2）时退回 GT 口径并标 ᴳᵀ。判定结论（4 胜/2 尺度/2 域界）不变，仅中心列换真值口径。",
          "",
          "> 2026-11-03 深夜 T9 纠偏（NC 审计判决实验 _nc_ref_duel/_nc_duel2/_nc_duel3.py）：**上段\"ceria GT 中心系统偏 15.5–17.9px\"系口径混案，撤回**——GT 中心两种口径：poni1/2÷px（naive）与 getFit2D()（带倾角换算），ceria 真倾角下差 ~8px；锐度对决显示 **GT-Fit2D 中心才是最锐处**（ceria_150 峰高 487 vs 共心 ref 158；E9M 20499 vs 8834；P2M ref≈GT）。修正：①GT 中心基本无辜，\"公开 GT 大面积错\"不成立；②concentric_center 的 σ（0.004–0.046px）是**精度不是准确度**——边切弧+倾角可致 ~6px 偏置；③E9M/mar 行 GTΔc 亦为 ref 偏置。审计 v0 表 docs/nc_audit_v0.md。",
          "> 2026-11-03 深夜终裁（_nc_duel4/_nc_duel5.py，经验半径法=无相/无λ/无D）：ceria_150 对照复现 GT-Fit2D 最锐；**mar3450 GT-Fit2D 同样最锐**——GTΔc=6.5 亦为 ref 偏置；halfccd 信噪低峰检无果（配置错配定性靠 T5 字节证据）。**最终格局**：倾角帧 GT-Fit2D 中心均最锐，平面解距最锐处 5–7px（fit_tilt 进中心路径）；无倾角帧 2Dⁿ 中心≈最优。",
          "> 2026-11-03 深夜 E1f（_nc_tilt_sharp.py）：平面解追加 fit_tilt（nsec=36 原生）后 ceria_150/250/300 Σ锐度 214/611/546→1648/1538/1133（GT 的 62/98/92%），tilt 解中心距 GT-Fit2D 仅 0.3–1.2px——5–7px 差距确系中心路径缺倾角。",
          "> 2026-11-03 深夜 E1f2（gui_solver.py 接线）：①native 帧 fit_tilt 改上原生图（px=px_m/s——池化画布把 imp 压 <5% 错杀真倾角）；②hybrid_ai V3 成对分支（D=解算 + tf poni/rot 自洽对；V1 只借 rot 实证 10× 更钝禁用）。ceria_150/250/300 判词 0.166/0.115/0.188°→**0.045/0.046/0.041°✔**、中心 6.65/5.05/25.2px→**1.24/0.34/0.52px**、ΔD 持平；ceria_150 悬案破案。验收：_t_spl_reg 重锁 12/12、phase_fit ceria 0.1/0.1mm、alumina 116.79/−0.07mm 不伤。",
          "> 2026-11-03 深夜 E1f3 A/B 臂（_nc_e1f3{,b}.py）：**E9M 中心 0.66px/锐度 87% GT**；**P2M 负对照守住**（imp 1% 门关）；Eiger4M 暴露非 native 帧的 V1 残伤（42px/4%）→ **V3 放开到所有 tilt_on 帧**：ceria_400 dev 0.095°→**0.024°**/Σ17×/中心 24.5→2.5px（悬案二号破案），Eiger4M Σ7×/中心 9.3px。验收三重全绿。",
          "> 2026-11-03 深夜 E1f3c（Δc(tilt) 列首上全表，canvas 口径）：mar3450 2.0px（旧 6.65px）、Eiger4M 9.3px、Pilatus1M 3.5px。教训入库：PS `>` 重定向把 stdout 写成 UTF-16+BOM；md 表格替换须先验证提取非空再写回。",
          "> 2026-11-03 深夜 E1f3d（zoo tilt 块上原生图：native 优先、canvas 兜底）：**全列脱胎换骨——6/6 可评行 Δc(tilt)≤2.4px、5 行亚像素**：Eiger4M **0.3px**（Δrot 距 GT 仅 (−1.0,−0.7)mrad，canvas 曾差 12mrad）、Eiger4M_v2 0.8px、**Eiger9M 0.7px（空列补齐）**、Pilatus1M 0.8px、P2M 0.9px（窄角域 Δrot 4mrad 属简并，生产门照拒）、mar 2.4px（17° 窄角域 rot 仍难、中心好）。另：本文件 md 列表为历史注唯一真源——手动补注会被下次运行冲掉（E1f3c/E1f3d 两次实证），注必须写进 zoo_run 内嵌列表。",
          "> 2026-11-05 R1/R2/R3（18-1104 需求）：画布策略与 fallback 门控统一进 solver_core.solve_image（默认档 max_side=520；升级档 ⌈maxdim/2⌉ 仅零样本安全门触发：a=默认档解不出 / b=rms>3.5 且 m<3 且升级档键序占优 / **c=s≥5 两段式——先廉价分叉预检（大画布单 λ 解带 auto_center），两档 D 相对分叉 ≥10% 才上 λ 网格救援，采用按 T4 键序 (ps,m,−rms)**。需求原案 (c) 的\"ps 提升≥0.05 且 m 不降\"被校准推翻：E4M/E9M 大档同解重解（D 分叉仅 0.2%/0.07% 但 Δps +0.15/+0.25）会被误劫持、mar 救援解 m=5<假解 11 会被误拒——依据 docs/unify_canvas_fallback.md + 探针 _r2_calib2/_r2_gate_test）。旧 T4 的\"相错 AND ps<0.55\"门依赖 GT 相签 → 已废；zoo_run.canvas_of(723) 与 _fallback_joint 删除。行值不变（本次回归 10/10 行逐格一致）：四胜行 P2M −0.05/E4M −0.03/v2 −0.10/E9M +0.01、mar3450 fallback 行 ΔD@λ̂ +5.87（c 门分叉 108%→λ̂=0.3611 网格点复产）。另修潜在 NameError：refine_lambda_d 此前未 import、异常被 try 吞掉（mar λ̂ 一直取网格点 0.966×λ₀ 而非 LSQ 收口点；修复后 σ=231% 照旧拒收、λ̂ 不变，行值零影响）。",
          "| 数据 | GT自检(相,n环,中位Δq/q) | 相盲鉴定 | GT D(mm) "
          "| ΔD(mm) | Δcenter(px) | GTΔc(px) | λ̂(Å) | ΔD@λ̂(mm) | Δrot/Δc(tilt) | 画布 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"] + rows
    out_base = "zoo_benchmark_v3"
    if lam_map is not None:
        # ---- T8 辅助模式：覆盖表头与输出名（零样本 v3 结构保持原样）----
        lam_lst = ", ".join(f"{k}={v}Å" for k, v in lam_map.items())
        md = [
            "# 跨仪器基准 · 辅助模式（用户给定 λ，T8）",
            "",
            f"生成：zoo_run.py --lam（λ 输入：{lam_lst}）｜ **口径：非零样本**——",
            "λ 为用户外部输入，用于以外部信息破 λ–D 简并（T6 定律：角域 ≲20°",
            "的图 λ 数学上不可辨，谁补信息都行）；判定列 ΔD = D@λu 对 GT 名义 D。",
            "**λ 的价值=λ 的精度（GIGO）**：给陈旧标称值无济于事。零样本表",
            "zoo_benchmark_v3.md 一字未动。机制：solve/fallback 全用 λu、",
            "λ̂ 搜索跳过、fallback 网格退化单点、T3/T7 精化照常。",
            "",
            "| 数据 | GT自检(相,n环,中位Δq/q) | 相盲鉴定 | GT D(mm) "
            "| ΔD@λu(mm) | Δcenter(px) | GTΔc(px) | λu(Å) | ΔD@λ̂ | Δrot(mrad) | 画布 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"] + rows
        out_base = "zoo_benchmark_assist"
    with open(os.path.join(ROOT, "docs", out_base + ".md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(ROOT, "docs", out_base + ".json"), "w",
              encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=1)
    print(f"\n→ docs/{out_base}.md / .json  ({len(rows)} rows)")


def _parse_lam(argv):
    """['--lam', 'Tag=0.3738,Tag2=0.92'] → dict；无 --lam 返回 None。"""
    lam_map = None
    for i, a in enumerate(argv):
        if a == "--lam" and i + 1 < len(argv):
            lam_map = {}
            for kv in argv[i + 1].split(","):
                k, _, v = kv.partition("=")
                lam_map[k.strip()] = float(v)
            break
    return lam_map


if __name__ == "__main__":
    _l = _parse_lam(sys.argv[1:])
    _pos = [a for i, a in enumerate(sys.argv[1:])
            if not (a == "--lam" or (i > 0
                                     and sys.argv[1:][i - 1] == "--lam"))]
    main(_pos[0] if _pos else None, lam_map=_l)
