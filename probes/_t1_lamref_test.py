# -*- coding: utf-8 -*-
"""_t1_lamref_test.py：T1 λ 联合精化验收探针（16 号文档 §4 T1）。

验收（2026-11-01 深夜跑）：
  Pilatus1M（默认画布）：λ̂ 与 1.0326 差 <0.5%；D@λ̂ 与 GT 1583.23 差
    <5mm；相 AgBh。
  mar3450（s=3 画布，max_side=1150）：λ̂ 与 0.370–0.371 差 <0.5%；
    D@λ̂ 与 GT 222.55 差 <2mm；相 LaB6（默认 s=7 画布仍不行，T4 处理）。

口径：λ̂/D@λ̂ 全零样本（GT 只用于评估）；D@λ̂ 对 GT 名义 D；
像素一致性 r_our/r_gt 同用 λ̂（与 zoo_run 同式）。
"""
import os
import sys
import time

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import fabio
import pyFAI.azimuthalIntegrator
from torchfai.phases import phase_qs
from solver_core import solve_auto_lambda, to_canvas


def load_img(path):
    img = np.asarray(fabio.open(path).data, dtype=np.float64)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    return img


def run_case(img_f, poni_f, max_side, lam_expect, d_tol_mm, label):
    print(f"\n===== {label} =====")
    img = load_img(os.path.join(ZOO, img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(
        os.path.join(ZOO, poni_f))
    lam0 = ai.wavelength * 1e10
    gt_D = ai.dist * 1000.0
    canvas, s, _, _ = to_canvas(img, max_side=max_side)
    px_m = float(ai.detector.pixel1) * s
    print(f"画布 {canvas.shape} (s={s})  名义λ={lam0:.4f}Å  GT D={gt_D:.2f}mm"
          f"  期望λ̂≈{lam_expect}")

    t0 = time.perf_counter()
    r = solve_auto_lambda(
        canvas, px_m, lam_lo=lam0 * 0.90, lam_hi=lam0 * 1.10, n_s=5,
        auto_center=True, refine_stages=((0.04, 7), (0.008, 7)), full=True)
    dt = time.perf_counter() - t0
    if r is None:
        print(f"[FAIL] 联合求解无解  ({dt:.1f}s)")
        return False
    ph, sol, lam_hat = r["phase"], r["sol"], r["lam"]
    cx, cy, D_px, m, rms, icorr, ps = sol[:7]
    D_lam = D_px * px_m * 1000.0
    dD = D_lam - gt_D
    lam_err = abs(lam_hat / lam_expect - 1.0)

    # 像素一致性（同 zoo_run 式：r_our vs r_gt，同 λ̂ 同 px）
    qs = np.asarray(phase_qs(ph), float)
    tt = 2.0 * np.arcsin(np.clip(lam_hat * qs / (4.0 * np.pi), 0, 0.9999))
    r_our = (D_lam / 1000.0) * np.tan(tt) / px_m
    r_gt = (gt_D / 1000.0) * np.tan(tt) / px_m
    vis = (r_gt > 25) & (r_gt < 0.5 * max(canvas.shape))
    dr = np.abs(r_our[vis] - r_gt[vis])

    try:
        f2d = ai.getFit2D()
        gt_cx, gt_cy = f2d["centerX"] / s, f2d["centerY"] / s
        dcp = float(np.hypot(cx - gt_cx, cy - gt_cy))
    except Exception:
        dcp = float("nan")

    print(f"λ̂={lam_hat:.4f}Å (期望 {lam_expect}, 偏差 {lam_err*100:.3f}%)"
          f"  相={ph}  m={m}  rms={rms:.2f}  ps={ps:+.3f}")
    print(f"D@λ̂={D_lam:.2f}mm  ΔD={dD:+.2f}mm  Δcenter={dcp:.1f}px(画布)"
          f"  一致性中位 {np.median(dr):.2f}/最大 {dr.max():.2f}px"
          f"  [{dt:.1f}s, {r['n_eval']}次评估]")
    for tp in r["top3"]:
        t2 = 2.0 * np.arcsin(np.clip(
            tp[2] * np.asarray(phase_qs(tp[0]), float) / (4.0 * np.pi),
            0, 0.9999))
        print(f"  粗网格top3: {tp[0]:8s} λ={tp[2]:.4f} m={tp[1][3]} "
              f"rms={tp[1][4]:.2f} ps={tp[1][6]:+.3f} "
              f"D={tp[1][2]*px_m*1000:.1f}mm")

    ok_lam = lam_err < 0.005
    ok_d = abs(dD) < d_tol_mm
    ok_ph = (ph == "AgBh") if "Pilatus1M" in label else (ph == "LaB6")
    print(f"验收: λ̂ {'✓' if ok_lam else '✗'}(<0.5%)  "
          f"D@λ̂ {'✓' if ok_d else '✗'}(<{d_tol_mm}mm)  "
          f"相 {'✓' if ok_ph else '✗'}({ph})")
    return ok_lam and ok_d and ok_ph


if __name__ == "__main__":
    r1 = run_case("Pilatus1M.edf", "Pilatus1M.poni", 520, 1.0326, 5.0,
                  "Pilatus1M 默认画布")
    r2 = run_case("LaB6_260210.mar3450", "Mar3450.poni", 1150, 0.3705, 2.0,
                  "mar3450 s=3 画布(max_side=1150)")
    print(f"\n==== T1 验收总结: Pilatus1M {'PASS' if r1 else 'FAIL'} / "
          f"mar3450(s3) {'PASS' if r2 else 'FAIL'} ====")
