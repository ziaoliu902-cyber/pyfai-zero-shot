# -*- coding: utf-8 -*-
"""_b8_distort_probe.py:B8 扭曲判决探针(2026-11-07)。

问题:探测器几何扭曲(spline 未校正)在 fit_tilt 残差里留方位角谐波
结构;生产需要一个零 GT 判决旗「建议 spline 校正」。

注入:ceria_200 干净帧 + spline 位移场(后向采样)→ 合成畸变帧。
幅度:阴性(×0)/ distorsion_2x2(弱)/ halfccd×0.5(中)/ halfccd(强)。

判决:fit_tilt 控制点 5 参残差 e_i=tth−fwd(p5);扇区 6 折交叉验证:
基线=test 扣 train 加权均值 vs 谐波模型 [1,cos2φ,sin2φ,cos4φ,sin4φ,
(r/rmax)²];reduction=1−rms_M/rms_base。阴性≈0、注入显著>0 即可分。
"""
import os
import sys

import numpy as np
import fabio
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc
from gui_solver import load_any
from poni_out import fit_tilt, make_pyfai_fwd
from torchfai.phases import phase_qs
from pyFAI.spline import Spline

LAM = 0.972386
PX_UM = 172.0
Q = phase_qs("CeO2")[:8]


def inject(img, sp_path, scale=1.0):
    """后向采样注入:I_dist[u] = I_clean[u − d(u)](d=spline 位移场×scale)。"""
    sp = Spline(sp_path)
    H, W = img.shape
    ys, xs = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float),
                         indexing="ij")
    dx = sp.splineFuncX(xs, ys) * scale
    dy = sp.splineFuncY(xs, ys) * scale
    m = np.isfinite(dx) & np.isfinite(dy)
    return map_coordinates(img, [np.where(m, ys - dy, 0.0),
                                 np.where(m, xs - dx, 0.0)],
                           order=1, mode="nearest"), \
        float(np.abs(dx[m]).max()) if m.any() else 0.0


def control_points(img):
    """solve_image → 原生几何 → fit_tilt 控制点。返回 (do, tf, D_mm) 或 None。"""
    r = sc.solve_image(img, PX_UM * 1e-6, lam=LAM)
    if r is None:
        return None
    cx, cy, D_px = r["sol"][0] * r["s"], r["sol"][1] * r["s"], r["sol"][2] * r["s"]
    do = {}
    tf = fit_tilt(img, cx, cy, D_px, Q, LAM, PX_UM * 1e-6, PX_UM * 1e-6,
                  sfull=1, nsec=36, mass_rel=0.10, bump=1.5, data_out=do)
    if tf is None:
        return None
    return do, tf, D_px * PX_UM * 1e-3


def cv_verdict(do, folds=6, seed=0):
    """扇区折 CV:谐波模型 vs 常数基线。返回 (reduction, base_rms, npts)。"""
    from numpy.polynomial import polynomial as P
    d1, d2, tth, w = do["d1"], do["d2"], do["tth"], do["w"]
    sec = do["sec"]
    p0 = np.asarray(do["p0"], float)
    fwd = make_pyfai_fwd(PX_UM * 1e-6, PX_UM * 1e-6, None)
    pred = fwd(p0[0], p0[1], p0[2], p0[3], p0[4], d1, d2)
    e = (tth - pred) * w                       # 加权残差
    phi = np.arctan2(d2 - do["poni2_0"], d1 - do["poni1_0"])
    r = np.hypot(d1 - do["poni1_0"], d2 - do["poni2_0"])
    rmax = r.max() if len(r) else 1.0
    A = np.column_stack([np.ones_like(phi), np.cos(2 * phi), np.sin(2 * phi),
                         np.cos(4 * phi), np.sin(4 * phi), (r / rmax) ** 2])
    secs = np.unique(sec)
    rng = np.random.default_rng(seed)
    rng.shuffle(secs)
    parts = np.array_split(secs, folds)
    base2, mod2, ntot = 0.0, 0.0, 0
    for held in parts:
        te = np.isin(sec, held)
        tr = ~te
        if te.sum() < 10 or tr.sum() < 30:
            continue
        c0 = float(np.average(e[tr], weights=np.ones_like(e[tr])))
        # 鲁棒化:Huber 损失拟合模型(RANSAC 精神,扇区级留出防过拟合)
        fit = least_squares(lambda c: (A[tr] @ c - e[tr]), np.zeros(A.shape[1]),
                            loss="soft_l1", f_scale=float(np.std(e[tr])))
        base2 += float(((e[te] - c0) ** 2).sum())
        mod2 += float(((e[te] - A[te] @ fit.x) ** 2).sum())
        ntot += int(te.sum())
    if ntot == 0:
        return 0.0, 0.0, 0
    base_rms = float(np.sqrt(base2 / ntot))
    mod_rms = float(np.sqrt(mod2 / ntot))
    return 1.0 - mod_rms / max(base_rms, 1e-12), base_rms, ntot


def _main():
    print("=" * 72)
    print("B8 扭曲判决探针:ceria_200 + spline 注入 × 幅度分级")
    print("=" * 72)
    base_img = load_any(ROOT + r"\data\real\MX_ceria\ceria_200_1_0001.cbf")

    CASES = [
        ("阴性 ×0(无注入)", None, 0.0),
        ("弱 distorsion_2x2", ROOT + r"\data\real\alumina\distorsion_2x2.spline", 1.0),
        ("中 halfccd×0.5", ROOT + r"\data\real\silx_zoo\halfccd.spline", 0.5),
        ("强 halfccd×1", ROOT + r"\data\real\silx_zoo\halfccd.spline", 1.0),
    ]
    rows = []
    for tag, sp_path, scale in CASES:
        if sp_path is None:
            img = base_img
            amp = 0.0
        else:
            img, amp = inject(base_img, sp_path, scale)
        got = control_points(img)
        if got is None:
            print(f"{tag:24} 解/倾角失败")
            rows.append((tag, amp, None, None))
            continue
        do, tf, D_mm = got
        red, base_rms, npts = cv_verdict(do)
        print(f"{tag:24} 注入幅 {amp:5.1f}px  D={D_mm:7.2f}mm  "
              f"控制点 {npts:4}  base_rms={base_rms*1e3:6.2f}mrad  "
              f"reduction={red:5.2%}")
        rows.append((tag, amp, red, base_rms))

    print("\n判决口径:reduction=扇区折外 1−rms_谐波/rms_常数")
    neg = [r for r in rows if r[2] is not None and r[1] == 0.0]
    pos = [r for r in rows if r[2] is not None and r[1] > 0.0]
    if neg and pos:
        n_max = max(r[2] for r in neg)
        p_min = min(r[2] for r in pos)
        thr = max(2 * n_max, 0.05)
        verdict = "✓ 阈值建议 reduction>%.0f%%" % (100 * thr) \
            if p_min > thr else "✗ 交叠,看上表"
        print(f"阴性 max={n_max:.2%}  阳性 min={p_min:.2%}  → 可分性 {verdict}")


if __name__ == "__main__":
    _main()
