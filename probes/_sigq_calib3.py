# -*- coding: utf-8 -*-
"""_sigq_calib3.py：洛伦兹拟合版环宽测量标定（第三版）。
v2 教训：滤波估背景有尺度困境（宽环被当背景吃掉/窄环贴地板）。
v3：find_peaks 定峰 → 窗内拟合 A·s²/((x-x0)²+s²) + 线性基线 —— 基线
自带扣背景，s（HWHM，bin 单位）即环宽，亚 bin 精度，无尺度参数。
判据：同几何 σq×10 → w 单调、比值>3；真实帧 w 落 sim 孪生域内。
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
import pyFAI
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from real_refine_v6 import to_nn_input_wide
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

LAM = 0.972386
K = 2.425
PXEFF = 172e-6 / (521.0 / 2527)
SIGS = [0.006, 0.010, 0.020, 0.030, 0.045, 0.060]
NB = 2048
MIN_R = 25.0
TOPK = 3


def _lor(x, A, x0, s, c, m):
    return A * s * s / ((x - x0) ** 2 + s * s) + c + m * x


def rel_width3(canvas, pr, pc):
    H, W = canvas.shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    r = np.hypot(yy - pr, xx - pc)
    rmax = float(r.max())
    idx = (r / rmax * (NB - 1)).astype(np.int32).ravel()
    v = canvas.astype(np.float64).ravel()
    cnt = np.bincount(idx, minlength=NB)
    prof = np.bincount(idx, weights=v, minlength=NB) / np.maximum(cnt, 1)
    pk, _ = find_peaks(prof, prominence=0.1 * prof.max(), distance=8)
    if len(pk) == 0:
        return np.nan
    top = pk[np.argsort(prof[pk])[-TOPK:]]
    bin_px = rmax / NB
    ws = []
    for p in top:
        r_px = (p + 0.5) * bin_px
        if r_px < MIN_R:
            continue
        w0 = max(2, int(0.08 * r_px / bin_px))     # 窗=预估环宽8倍(0.08r)
        lo, hi = max(0, p - w0), min(NB, p + w0 + 1)
        x = np.arange(lo, hi, dtype=np.float64)
        y = prof[lo:hi]
        try:
            popt, _ = curve_fit(
                _lor, x, y,
                p0=[prof[p] - prof[max(lo, p - w0)], float(p), w0 / 4.0,
                    prof[max(lo, p - w0)], 0.0],
                maxfev=2000)
            A, x0, s = popt[0], popt[1], abs(popt[2])
            if A > 0 and 0.3 < s < w0:                # 有效拟合
                ws.append(2 * s * bin_px / r_px)      # FWHM/r
        except Exception:
            continue
    return float(np.median(ws)) if ws else np.nan


print("v3 洛伦兹拟合版标定：")
for d in (250, 450):
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    db = get_scaled_detector("Pilatus_1M", 2, K)
    Hm, Wm = db.extent
    pr = p.poni1 / PXEFF
    pc = p.poni2 / PXEFF
    g_geom = dict(dist=p.dist, poni1=Hm * (pr / 521), poni2=Wm * (pc / 490),
                  rot1=p.rot1, rot2=p.rot2, rot3=p.rot3)
    print(f"-- d={d} --")
    for sg in SIGS:
        rng = np.random.default_rng(7)
        cfg = SimConfig()
        cfg.wavelength = LAM * 1e-10
        cfg.beamstop_prob = 0.0
        cfg.peak_sigma_range = (sg, sg)
        cfg.peak_sigma_broad_frac = 0.0
        s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                         pixel_scale=K, geom=g_geom)
        w = rel_width3(s["img"].astype(np.float64), pr, pc)
        print(f"{sg:7.3f}{w:10.4f}")

print("\n真实帧 w（v3）：")
for d in (150, 250, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    wide, k, extra = to_nn_input_wide(img)
    w = rel_width3(wide.astype(np.float64), p.poni1 / PXEFF,
                   (p.poni2 - extra) / PXEFF)
    print(f"ceria{d}: w={w:.4f}")
