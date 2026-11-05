# -*- coding: utf-8 -*-
"""_sigq_calib2.py：背景扣除版环宽测量标定。
v1 污染源诊断：径向谱含背景台阶（bg0+bg1·q+驼峰），半高游走被台阶抬高
→ 同几何动态范围仅 1.27×。v2：中值滤波(101bin)估背景 → 净轮廓量 FWHM。
判据：同几何 σq×10 → w 单调且比值>3；真实帧 w 落 sim 孪生域内。
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
from scipy.ndimage import median_filter
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
TOPK = 4


def rel_width2(canvas, pr, pc):
    H, W = canvas.shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    r = np.hypot(yy - pr, xx - pc)
    rmax = float(r.max())
    idx = (r / rmax * (NB - 1)).astype(np.int32).ravel()
    v = canvas.astype(np.float64).ravel()
    cnt = np.bincount(idx, minlength=NB)
    prof = np.bincount(idx, weights=v, minlength=NB) / np.maximum(cnt, 1)
    bg = median_filter(prof, size=101, mode="nearest")   # 慢变背景（环<10px）
    net = np.clip(prof - bg, 0, None)
    pk, _ = find_peaks(net, prominence=0.15 * net.max(), distance=8)
    if len(pk) == 0:
        return np.nan
    top = pk[np.argsort(net[pk])[-TOPK:]]
    ws = []
    bin_px = rmax / NB
    for p in top:
        r_px = (p + 0.5) * bin_px
        if r_px < MIN_R:
            continue
        half = net[p] * 0.5
        i = p
        while i > 0 and net[i] > half:
            i -= 1
        j = p
        while j < NB - 1 and net[j] > half:
            j += 1
        ws.append((j - i) * bin_px / r_px)
    return float(np.median(ws)) if ws else np.nan


print("v2 背景扣除版标定：")
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
        w = rel_width2(s["img"].astype(np.float64), pr, pc)
        print(f"{sg:7.3f}{w:10.4f}")

print("\n真实帧 w（v2）：")
for d in (150, 250, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    wide, k, extra = to_nn_input_wide(img)
    w = rel_width2(wide.astype(np.float64), p.poni1 / PXEFF,
                   (p.poni2 - extra) / PXEFF)
    print(f"ceria{d}: w={w:.4f}")
