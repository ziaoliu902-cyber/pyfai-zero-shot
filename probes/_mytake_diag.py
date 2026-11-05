# -*- coding: utf-8 -*-
"""独立复核 zoo 失败项：Pilatus1M / mar3450 / Pilatus6M。

不复用 zoo_run.main（避免覆写文档），只借它的载入/自检/画布函数，
外加自己的取证：环完整性、Kasa 定心偏移、GT poni 的径向峰表。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import pyFAI.azimuthalIntegrator
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

import zoo_run as ZR
import solver_core as SC
from torchfai.phases import PHASE_NAMES, phase_qs


def ring_peaks_1d(img, ai, npt=4096):
    q, I = ai.integrate1d(img, npt, unit="q_A^-1")
    I = np.nan_to_num(np.asarray(I, float))
    bg = median_filter(I, 121)
    d = I - bg
    m = q > 0.05
    pk, props = find_peaks(d, prominence=0.04 * d[m].max(), distance=6)
    keep = m[pk]
    return q[pk[keep]], d[pk[keep]], q, I


def completeness(img, cx, cy, r_pred, ring_hw=3.0, n_az=72):
    """GT 中心上，每条预测环半径的方位角覆盖率与峰值显著度。"""
    from scipy.ndimage import map_coordinates
    H, W = img.shape
    th = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    out = []
    for r in r_pred:
        if r < 5 or r > min(H, W) / 2 * 1.05:
            continue
        vals = []
        for t in th:
            x = cx + r * np.cos(t)
            y = cy + r * np.sin(t)
            if 1 <= x < W - 2 and 1 <= y < H - 2:
                vals.append(map_coordinates(img, [[y], [x]], order=1)[0])
        if len(vals) < n_az * 0.5:
            out.append((r, 0.0, np.nan))
            continue
        v = np.asarray(vals)
        bg = np.median(v)
        sig = (np.percentile(v, 90) - bg) / (np.std(v) + 1e-9)
        # 方位角覆盖：有多少扇区的亮度显著高于背景
        out.append((r, len(v) / n_az, sig))
    return out


def report(tag, img_f, poni_f):
    print("=" * 78)
    print(f"### {tag}")
    img = ZR.load_img(os.path.join(ZOO, img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    print("shape", img.shape, "detector", ai.detector.name,
          "pixel(um)", ai.detector.pixel1 * 1e6, ai.detector.pixel2 * 1e6,
          "dist(mm)", ai.dist * 1000, "lambda(A)", (ai.wavelength or 0) * 1e10)
    try:
        f2d = ai.getFit2D()
        print(f"GT center (px): x={f2d['centerX']:.2f} y={f2d['centerY']:.2f}"
              f"  rot1/2(mrad)={ai.rot1*1000:.2f},{ai.rot2*1000:.2f}")
        gx, gy = f2d["centerX"], f2d["centerY"]
    except Exception:
        gx, gy = ai.poni2 / ai.detector.pixel2, ai.poni1 / ai.detector.pixel1
        print("GT center (px, fallback):", gx, gy)

    ph, n, med, ok, s0 = ZR.gt_selfcheck(img, ai)
    print(f"GT selfcheck: phase={ph} n={n} med={med*100:.2f}% ok={ok} scale={s0:.4f}")

    qpk, dpk, qax, I = ring_peaks_1d(img, ai)
    print(f"1D peaks (GT center): {len(qpk)}  q=[{qpk.min():.3f},{qpk.max():.3f}]")
    if ok and ph:
        qs = np.asarray(phase_qs(ph), float)
        src = np.abs(qpk[:, None] / qs[None, :] - 1.0).min(axis=1)
        print(f"  peaks within 1.2% of {ph} lines: {int((src<0.012).sum())}/{len(qpk)}")

    # 环完整性（GT 中心）
    if ok and ph:
        lam = ai.wavelength
        tt = 2 * np.arcsin(np.clip(lam * np.asarray(phase_qs(ph)) / (4 * np.pi), 0, 0.9999))
        r_pred = ai.dist * np.tan(tt) / ai.detector.pixel1
        comp = completeness(img, gx, gy, r_pred)
        print("  ring completeness (r_px, azimuth coverage, contrast):")
        for r, cov, sig in comp:
            print(f"    r={r:7.1f}  cov={cov:.2f}  contrast={sig:.1f}")

    # 画布 + Kasa 定心（从 GT 中心出发）看看偏多少
    canvas, s, cx0, cy0 = ZR.to_canvas(img)
    print(f"canvas {canvas.shape} s={s} center_px=({cx0:.1f},{cy0:.1f})")
    try:
        k = SC.kasa_center(canvas, (gx / s, gy / s))
        print("kasa_center from GT:", None if k is None else
              tuple(round(v, 2) for v in k))
    except Exception as e:
        print("kasa_center failed:", e)

    # 锐度定心
    try:
        sh = SC.center_by_sharpness(canvas, dbg=False)
        print("center_by_sharpness:", None if sh is None else
              tuple(round(v, 2) for v in sh[:2]))
    except Exception as e:
        print("center_by_sharpness failed:", e)

    # 零样本求解（生产路径）
    import time
    lam_A = (ai.wavelength or 1e-10) * 1e10
    t0 = time.perf_counter()
    r = SC.solve_auto(canvas, lam_A, ai.detector.pixel1 * s, cascade=True)
    dt = (time.perf_counter() - t0) * 1000
    if r is None:
        print("solve_auto: None")
        return
    ph_fit, sol = r
    cx, cy, D_px, m, rms, icorr, pscore = sol[:7]
    D_fit = D_px * ai.detector.pixel1 * s * 1000
    print(f"solve_auto: phase={ph_fit} D={D_fit:.2f}mm (GT {ai.dist*1000:.2f}, "
          f"dD={D_fit-ai.dist*1000:+.2f}) m={m} rms={rms:.2f} ps={pscore:+.3f} "
          f"center=({cx*s:.1f},{cy*s:.1f}) GT=({gx:.1f},{gy:.1f}) "
          f"dc={np.hypot(cx*s-gx, cy*s-gy):.1f}px  {dt:.0f}ms")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    items = {
        "Pilatus1M": ("Pilatus1M.edf", "Pilatus1M.poni"),
        "mar3450": ("LaB6_260210.mar3450", "Mar3450.poni"),
        "Pilatus6M": ("Pilatus6M.cbf", "Pilatus6M.poni"),
    }
    for k, (img_f, poni_f) in items.items():
        if which in ("all", k):
            report(k, img_f, poni_f)
