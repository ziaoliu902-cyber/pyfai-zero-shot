# -*- coding: utf-8 -*-
"""pyfai_duel：同一问题，pyFAI 标准流程 vs 我们，三臂对决（2026-11-05）。

"pyFAI 对照"= 同图 + 同检测器(含 spline) + 同 λ(GT poni) + 同标样身份，
让 pyFAI 用它自己的标准自动化流程（RingExtraction 找环指派 +
GeometryRefinement.refine3，即 pyFAI-calib2 无 GUI 内核）解一遍，比准。

三臂：
  T  = 我们：solve_image 统一入口（默认档 520 + R2 零样本 fallback）+
      T7 原生精化（zoo 标准行口径）；ceria 行=锁表 8 环生产口径。
  P0 = pyFAI 零起点：中心=几何中心，D∈{0.1,0.3,1.0}m 网格（替它试三
      个起点，取它自己 chi2 最优——不看 GT）。
  P1 = pyFAI + 我们初值：同 P0 流程但以 T 解初始——"同起点，纯机器
      精修谁强"。

指标：ΔD(mm)、Δcenter(px) 双口径（GT / 共心真值 T9）、点数、耗时。
域界如实记：Frelon2k/halfccd 的 GT poni 带 spline，pyFAI 检测器吃到
spline 而我们不吃——预期它占优的行不遮掩。
输出：probes/results/pyfai_duel_v1.md + .json
"""
import os
import sys
import time
import json
import traceback

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import zoo_run                                    # noqa: E402（复用 load_img/ITEMS；chdir→ZOO）
from zoo_run import ITEMS, load_img               # noqa: E402

import pyFAI                                       # noqa: E402
from pyFAI.calibrant import get_calibrant          # noqa: E402
from pyFAI.goniometer import SingleGeometry        # noqa: E402
from pyFAI.ring_extraction import RingExtraction   # noqa: E402
from pyFAI.geometry import Geometry                # noqa: E402
from pyFAI.geometryRefinement import GeometryRefinement  # noqa: E402

from torchfai.phases import phase_qs               # noqa: E402
from solver_core import solve_image, concentric_center  # noqa: E402

# ---- pyFAI 2025.12 官方 bug 绕道：RingExtraction._calculate_num_of_
# points_to_keep 把 Unit 对象按位置参塞进 center_array(shape,...) 的
# shape 位 → numpy.indices 崩（'Unit' object is not iterable）。__init__
# 里同函数用 unit= 关键字传是对的。补丁=按签名正确传 shape（顺带修
# clip 越界 off-by-one）。上游修复前必须保留。----
def _npk_fixed(self, pixels_at_two_theta_level, points_per_degree):
    import pyFAI.units as _u
    sh = self.image.shape
    chi = self.single_geometry.geometry_refinement.center_array(
        sh, unit=_u.CHI_RAD, scale=False)
    vals = chi[pixels_at_two_theta_level[:, 0].clip(0, sh[0] - 1),
               pixels_at_two_theta_level[:, 1].clip(0, sh[1] - 1)]
    return int(np.unique(np.rad2deg(vals).round()).size * points_per_degree)


RingExtraction._calculate_num_of_points_to_keep = _npk_fixed

ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
CERIA = os.path.join(ROOT, "data", "real", "MX_ceria")
OUT_DIR = os.path.join(ROOT, "probes", "results")

# GT 自检通过相 → pyFAI 标样名（zoo 2026-11-05 口径；Al2O3 在 pyFAI
# 注册名是 alpha_Al2O3，quartz=quartz）
CAL_MAP = {"AgBh": "AgBh", "CeO2": "CeO2", "Al2O3": "alpha_Al2O3",
           "LaB6": "LaB6", "Quartz": "quartz"}
P0_GRID = (0.1, 0.3, 1.0)      # pyFAI 零起点 D 网格(m)
MAX_RINGS = 15


def gt_of(poni_path):
    ai = pyFAI.load(poni_path)
    try:
        f2d = ai.getFit2D()
        gx, gy = f2d["centerX"], f2d["centerY"]
    except Exception:
        gx = ai.poni2 / ai.detector.pixel2
        gy = ai.poni1 / ai.detector.pixel1
    return ai, gx, gy, ai.dist * 1000.0


def ours_run(img, px_native, lam_A, q_list=None, ph_guess=None):
    """T 臂：solve_image + T7 原生精化（zoo 标准行口径）。

    lam_A=None → 无 λ 联合模式（Frelon2k/halfccd：GT poni 无波长，
    zoo 同款分支）；此时 res["lam"]=λ̂。"""
    t0 = time.perf_counter()
    res = solve_image(img, px_native, lam=lam_A, q_list=q_list)
    if res is None:
        return None
    s, px_m = res["s"], res["px_m"]
    sol = list(res["sol"])
    ph_fit = res["phase"] or ph_guess
    lam_used = res["lam"]
    r2d_n = int(sol[10]) if len(sol) >= 11 else 0
    if r2d_n:
        try:
            from solver_core import refine_geometry_2d as rg2
            q7 = np.asarray(phase_qs(ph_fit), float)
            o = rg2(img, sol[0] * s, sol[1] * s, sol[2] * s, q7,
                    lam_used, n_iter=3)
            if o is not None:
                sol[0], sol[1], sol[2] = o[0] / s, o[1] / s, o[2] / s
        except Exception:
            pass
    cx, cy, D_px, m = sol[0], sol[1], sol[2], sol[3]
    return dict(cx=float(cx * s), cy=float(cy * s),
                D_mm=float(D_px * px_m * 1000.0), m=int(m),
                lam=float(lam_used) if lam_used else None,
                ms=(time.perf_counter() - t0) * 1000,
                fb=bool(res["used_fallback"]), fbr=str(res["fb_reason"]),
                tier=(f"s={s}" + ("·fb" if res["used_fallback"] else "")
                      + ("ⁿ" if r2d_n else "")))


def pyfai_run(img, det, calname, lam_A, D0, cx0, cy0):
    """P 臂：RingExtraction 自动找环指派 + GeometryRefinement.refine3，
    两轮（refine → 以精化几何重提取 → refine——calib2"Recalibrate"迭代
    的无 GUI 等价）。

    初值 = (D0[m], cx0/cy0[原生px])。lam_A=None → λ 联合模式（GT 无波
    长，Frelon/halfccd）：λ 初值 1.0Å、refine3 不 fix wavelength（pyFAI
    自己的联合 λ–D 精修）。返回 dict 或 None（带 err 字段时为失败原因）。"""
    lam_m = lam_A * 1e-10 if lam_A else 1.0e-10
    t0 = time.perf_counter()
    try:
        cal = get_calibrant(calname)
        cal.wavelength = lam_m

        def _round(geo_in):
            sg = SingleGeometry("duel", image=img, calibrant=cal,
                                detector=det, geometry=geo_in)
            sg.set_wavelength(lam_m)
            rex = RingExtraction(sg)
            cps = rex.extract_control_points(max_number_of_rings=MAX_RINGS)
            data = cps.getWeightedList(img)
            if data is None or len(data) < 6:
                return None, 0
            gr = GeometryRefinement(
                data=data, calibrant=cal, detector=det, wavelength=lam_m,
                dist=geo_in.dist, poni1=geo_in.poni1, poni2=geo_in.poni2,
                rot1=geo_in.rot1, rot2=geo_in.rot2, rot3=geo_in.rot3)
            gr.refine3(fix=(["wavelength"] if lam_A else []))
            return gr, int(len(data))

        geo = Geometry(dist=D0, poni1=cy0 * det.pixel1,
                       poni2=cx0 * det.pixel2,
                       rot1=0.0, rot2=0.0, rot3=0.0, wavelength=lam_m,
                       detector=det)
        gr, npts = _round(geo)
        if gr is None:
            return dict(err=f"控制点不足({npts})",
                        ms=(time.perf_counter() - t0) * 1000)
        npts2 = 0
        try:                            # 第二轮：以精化几何重提取再精修
            gr2, npts2 = _round(gr.__class__.copy(gr)
                                if hasattr(gr, "copy") else gr)
            if gr2 is not None:
                gr, npts = gr2, (npts2 or npts)
        except Exception:
            pass
        try:
            chi2 = float(gr.chi2())
        except Exception:
            chi2 = None
        return dict(cx=float(gr.poni2 / det.pixel2),
                    cy=float(gr.poni1 / det.pixel1),
                    D_mm=float(gr.dist * 1000.0), npts=int(npts),
                    lam=float(gr.wavelength * 1e10),
                    chi2=chi2, ms=(time.perf_counter() - t0) * 1000)
    except Exception:
        return dict(err=traceback.format_exc(limit=2)[-160:].replace("\n", " "),
                    ms=(time.perf_counter() - t0) * 1000)


def cref_dc(img, cx, cy, D_mm, ph, lam_A, px_native):
    """共心真值口径 Δcenter（T9；GT 无关，双臂同一把尺）。"""
    try:
        qs = np.asarray(phase_qs(ph), float)
        r = concentric_center(img, (cx, cy), D_mm / 1000.0 / px_native,
                              qs, lam_A)
        if r is None:
            return None
        return float(np.hypot(cx - r["cx"], cy - r["cy"]))
    except Exception:
        return None


def duel(tag, img_path, poni_path, calname, ours_q_list=None,
         ours_ph_guess=None):
    img = load_img(img_path)
    ai, gx, gy, gD = gt_of(poni_path)
    det = ai.detector
    px_native = float(det.pixel1)
    # Frelon2k/halfccd GT poni 无 wavelength → λ 联合模式（T 走 zoo 无λ
    # 分支 solve_image(lam=None)；P 臂 λ 初值 1.0Å 放开精修）
    lam_A = float(ai.wavelength) * 1e10 if ai.wavelength else None
    out = dict(tag=tag, cal=calname, gt_D=gD, gt_c=[gx, gy],
               shape=list(img.shape), lam_gt=lam_A)

    # ---- T ----
    try:
        t = ours_run(img, px_native, lam_A, q_list=ours_q_list,
                     ph_guess=ours_ph_guess)
    except Exception:
        t = dict(err=traceback.format_exc(limit=2)[-120:].replace("\n", " "))
    out["T"] = t
    if t and "cx" in t:
        t["dc_gt"] = float(np.hypot(t["cx"] - gx, t["cy"] - gy))
        t["dc_ref"] = cref_dc(img, t["cx"], t["cy"], t["D_mm"],
                              ours_ph_guess or "CeO2",
                              t.get("lam") or lam_A, px_native)

    # ---- P0：零起点网格，取 pyFAI 自己 chi2 最优 ----
    H, W = img.shape
    best, best_chi2 = None, None
    for d0 in P0_GRID:
        r = pyfai_run(img, det, calname, lam_A, d0, W / 2.0, H / 2.0)
        if r is None or "cx" not in r:
            continue
        if best_chi2 is None or (r["chi2"] is not None
                                 and r["chi2"] < best_chi2):
            best, best_chi2 = r, r["chi2"]
    out["P0"] = best if best is not None else dict(err="全网格失败")
    if best and "cx" in best:
        best["dc_gt"] = float(np.hypot(best["cx"] - gx, best["cy"] - gy))
        best["dc_ref"] = cref_dc(img, best["cx"], best["cy"], best["D_mm"],
                                 ours_ph_guess or "CeO2",
                                 best.get("lam") or lam_A, px_native)

    # ---- P1：以 T 解为初值（λ 联合模式 λ 初值仍 1.0Å，只给 D/中心）----
    if t and "cx" in t:
        p1 = pyfai_run(img, det, calname, lam_A, t["D_mm"] / 1000.0,
                       t["cx"], t["cy"])
        if "cx" in p1:
            p1["dc_gt"] = float(np.hypot(p1["cx"] - gx, p1["cy"] - gy))
            p1["dc_ref"] = cref_dc(img, p1["cx"], p1["cy"], p1["D_mm"],
                                   ours_ph_guess or "CeO2",
                                   p1.get("lam") or lam_A, px_native)
        out["P1"] = p1
    else:
        out["P1"] = dict(err="T 失败，无初值")
    return out, img, (gx, gy, gD), lam_A, px_native


def lamd_pct(r, gt_D, lam_gt):
    """λD 简并不变量相对误差(%)：λ–D 脊上≈0（几何对、λ̂ 吸收标称差）。
    λ 联合行（GT 无 λ）无分母 → None。"""
    if not lam_gt or not r or "D_mm" not in r:
        return None
    lam = r.get("lam") or lam_gt
    return (lam * r["D_mm"] / (lam_gt * gt_D) - 1.0) * 100.0


def cell(r, gt_D, lam_gt):
    """(ΔD, ΔλD%, Δcᴳᵀ, Δcᶜᵒⁿ, λ̂, 点) 六元组字符串。"""
    if r is None or "cx" not in r:
        return [f"FAIL:{(r or {}).get('err', '?')[:24]}"] + ["—"] * 5
    dd = r["D_mm"] - gt_D
    lp = lamd_pct(r, gt_D, lam_gt)
    lps = "—" if lp is None else f"{lp:+.2f}"
    dcr = "—" if r.get("dc_ref") is None else f"{r['dc_ref']:.1f}"
    lam = r.get("lam")
    return [f"{dd:+.2f}", lps, f"{r['dc_gt']:.1f}", dcr,
            f"{lam:.4f}" if lam else "—",
            str(r.get("npts", r.get("m", "—")))]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = []          # (tag, img_path, poni, calname, q_list, ph_guess)
    for fn, ponis in ITEMS:
        base = os.path.splitext(fn)[0]
        img_path = os.path.join(ZOO, fn)
        if not os.path.exists(img_path):
            continue
        for p in ponis:
            tag = base if len(ponis) == 1 else base + "_" + p[:-5]
            # GT 相用 zoo 已自检结论（跑 gt_selfcheck 太贵，结论固定）
            ph = {"Pilatus1M": "AgBh", "Pilatus2M": "CeO2",
                  "Pilatus6M": None, "Eiger4M_Eiger4M": "Al2O3",
                  "Eiger4M_Eiger4M_v2": "Al2O3", "Eiger9M": "CeO2",
                  "Frelon2k": "Al2O3", "halfccd": "Al2O3",
                  "LaB6_260210": "LaB6", "Fairchild": None}.get(tag, None)
            if ph is None:
                continue          # P6M(非标定帧)/Fairchild(域界) 不进分母
            jobs.append((tag, img_path, os.path.join(ZOO, p),
                         CAL_MAP[ph], None, ph))
    # ceria 生产行（锁表 8 环口径=我们的生产实践；pyFAI 用它自己的
    # CeO2 全表标准实践）
    for d in (150, 250, 400):
        f = f"ceria_{d}_1_0001"
        jobs.append((f"ceria{d}", os.path.join(CERIA, f + ".cbf"),
                     os.path.join(CERIA, f + ".poni"), "CeO2",
                     phase_qs("CeO2")[:8], "CeO2"))

    rows, jout = [], []
    print(f"# pyfai_duel v2  {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"pyFAI {pyFAI.version}；三臂 T/P0/P1；P0 网格 {P0_GRID} m，"
          f"max_rings={MAX_RINGS}\n")
    for tag, ip, pp, cal, ql, phg in jobs:
        try:
            r, img, (gx, gy, gD), lam_A, pxn = duel(tag, ip, pp, cal, ql, phg)
        except Exception:
            rows.append(f"| {tag} | ERROR | `{traceback.format_exc(limit=2)[-140:]}` | | | | | | | |")
            jout.append(dict(tag=tag, error=True,
                             tb=traceback.format_exc(limit=4)))
            print("!", tag, "ERROR"); continue
        T, P0, P1, lam_gt = r["T"], r["P0"], r["P1"], r["lam_gt"]
        tier = T.get("tier", "?") if T else "?"
        fb = ("·λ̂fb" if T and T.get("fb") else "")
        cT, c0, c1 = (cell(T, gD, lam_gt), cell(P0, gD, lam_gt),
                      cell(P1, gD, lam_gt))
        rows.append(
            f"| {tag} | {cal} | T {tier}{fb} | {cT[0]} | {cT[1]} "
            f"| {cT[2]} | {cT[3]} | {cT[4]} | {cT[5]} |"
            f" {T.get('ms',0)/1000:.0f}s |")
        rows.append(
            f"|  |  | P0 | {c0[0]} | {c0[1]} "
            f"| {c0[2]} | {c0[3]} | {c0[4]} | {c0[5]} |"
            f" {P0.get('ms',0)/1000:.0f}s* |")
        rows.append(
            f"|  |  | P1 | {c1[0]} | {c1[1]} | {c1[2]} | {c1[3]} "
            f"| {c1[4]} | {c1[5]} | {P1.get('ms',0)/1000:.0f}s |")
        jout.append(r)
        print(f"[{tag}] T={cT[0]}/{cT[2]} P0={c0[0]}/{c0[2]} "
              f"P1={c1[0]}/{c1[2]}")

    md = ["# pyFAI 对照（duel v2）——同图同 λ 同标样，标准流程对决",
          "",
          f"pyFAI {pyFAI.version}；T=solve_image 统一入口+T7 原生精化"
          "（ceria 行锁表 8 环）；P0=pyFAI RingExtraction+refine3 零起点"
          f"（D 网格 {P0_GRID} m 取其 chi2 最优，中心=几何中心）；"
          "P1=同 P0 但以 T 解为初值。Frelon2k/halfccd GT poni 无波长 →"
          " λ 联合模式（T=联合 λ–D；P 臂 λ 初值 1Å 放开精修）。",
          "ΔD 单位 mm；ΔλD=λ̂·D̂ 相对 GT 的偏差%（λ–D 简并脊上≈0 表示"
          "几何正确、λ̂ 吸收了标称波长差）；Δc 单位原生 px（ᴳᵀ=GT 口径，"
          "ᶜᵒⁿ=共心真值 T9）。*P0 耗时=三网格总和。",
          "",
          "| 项 | 标样 | 臂 | ΔD(mm) | ΔλD(%) | Δcᴳᵀ | Δcᶜᵒⁿ | λ̂(Å) | 点/环 | 耗时 |",
          "|---|---|---|---|---|---|---|---|---|---|"] + rows
    with open(os.path.join(OUT_DIR, "pyfai_duel_v2.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_DIR, "pyfai_duel_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(jout, f, ensure_ascii=False, indent=1, default=str)
    print("\n".join(rows))
    print("\nsaved -> probes/results/pyfai_duel_v2.md/.json")


if __name__ == "__main__":
    main()
