# -*- coding: utf-8 -*-
"""外环衰减终审：
T1 修正版移植：孪生 × (真实径向剖面/孪生径向剖面)，无 0.05 底、宽 clip
T2 物理压暗：孪生 × exp(-B' q²)，B' 按真实 I(q4.02)/I(q2.01) 反解
预测若从 ~-10 掉到 ~-60 → 外环衰减 = 真凶（治=训练域放宽 dw_b_range）。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI

from real_refine_v6 import load_px_nn, match_gain, to_nn_input_wide
from torchfai.config import SimConfig
from torchfai.dataset import norm_to_param
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

m, dev, consts = load_px_nn(r"runs\v4w2\best.pt")
from torchfai.dataset import TorchFAIDataset as _T
consts = _T(os.path.join(ROOT, "data", "train_v4w2"), "val").norm_consts
m.eval()
LAM = 0.972386
K = 2.425
PXEFF = 172e-6 / (521.0 / 2527)
QS = np.array([2.011, 2.322, 3.284, 3.851, 4.022, 4.650, 5.061, 5.192])


def coarse(canvas, k):
    x = torch.from_numpy(
        np.log1p(canvas[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([LAM], device=dev),
                  torch.tensor([k], device=dev))
    return float(norm_to_param(mu.double().cpu().numpy(), consts, k)[0, 0])


def radial(canvas, pr, pc, nb=245):
    yy, xx = np.mgrid[0:521, 0:490]
    r = np.hypot(yy - pr, xx - pc)
    idx = np.clip((r / 2).astype(int), 0, nb - 1)
    prof = np.bincount(idx.ravel(), weights=canvas.ravel(), minlength=nb)
    cnt = np.bincount(idx.ravel(), minlength=nb)
    return prof / np.maximum(cnt, 1), idx


print(f"{'帧':>5s}{'真实Δ':>8s}{'孪生Δ':>8s}{'T1移植Δ':>9s}{'T2压暗Δ':>9s}"
      f"{'B_real':>8s}{'B_sim':>7s}")
for d in (250, 350, 400):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    D = p.dist
    wide, k, extra = to_nn_input_wide(img)
    pr = p.poni1 / PXEFF
    pc = (p.poni2 - extra) / PXEFF

    db = get_scaled_detector("Pilatus_1M", 2, K)
    Hm, Wm = db.extent
    rng = np.random.default_rng(7)
    g = dict(dist=D, poni1=Hm * (pr / 521), poni2=Wm * (pc / 490),
             rot1=0.0, rot2=0.0, rot3=0.0)
    cfg = SimConfig()
    cfg.wavelength = LAM * 1e-10
    cfg.beamstop_prob = 0.0
    s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                     pixel_scale=K, geom=g)
    twin = match_gain(s["img"].astype(np.float64))

    # 环半径 → q 桶映射（用于测 B）
    yy, xx = np.mgrid[0:521, 0:490]
    rr_px = np.hypot(yy - pr, xx - pc)
    r_m = rr_px * PXEFF
    two_th = np.arctan(r_m / D)
    qmap = 4 * np.pi / (LAM * 1e-10) * np.sin(two_th / 2)
    qbin = np.clip((qmap / 6.5 * 122).astype(int), 0, 121)

    def qprofile(canvas):
        prof = np.bincount(qbin.ravel(), weights=canvas.ravel(), minlength=122)
        cnt = np.bincount(qbin.ravel(), minlength=122)
        return prof / np.maximum(cnt, 1)

    qr, qt = qprofile(wide), qprofile(twin)

    def peakI(prof, q0, dq=0.06):
        j0 = int(q0 / 6.5 * 122)
        w = int(dq / 6.5 * 122)
        return prof[j0 - w:j0 + w + 1].max() if j0 > w else np.nan

    def B_est(prof):
        try:
            i1 = peakI(prof, 2.011)
            i4 = peakI(prof, 3.851)
            if not (np.isfinite(i1) and np.isfinite(i4)) or i1 <= 0 or i4 <= 0:
                return np.nan
            ratio = i4 / i1
            if ratio <= 0:
                return np.nan
            # I ∝ A·exp(-Bq²) → B = ln(I1/I4)/(q4²-q1²)
            return float(np.log(1 / ratio) / (3.851**2 - 2.011**2))
        except Exception:
            return np.nan

    B_r, B_t = B_est(qr), B_est(qt)

    b_real = (coarse(wide, k) - D) * 1000
    b_twin = (coarse(twin, k) - D) * 1000

    # ---- T1：无底线宽 clip 移植 ----
    prof_r, _ = radial(wide, pr, pc)
    prof_t, idx = radial(twin, pr, pc)
    ratio = prof_r / np.maximum(prof_t, 0.02 * prof_t.max())
    ratio = np.clip(ratio, 0.01, 10.0)
    trans = np.clip(twin * ratio[idx], 0, 65535)
    b_t1 = (coarse(trans, k) - D) * 1000

    # ---- T2：物理压暗 exp(-B'q²)，B' 用真实帧反解值 ----
    if np.isfinite(B_r):
        damp = np.exp(-B_r * qmap**2 / np.exp(-0.0 * qmap**2))
        crush = np.clip(twin * damp, 0, 65535)
        b_t2 = (coarse(crush, k) - D) * 1000
    else:
        b_t2 = float("nan")

    print(f"{d:>5d}{b_real:+8.1f}{b_twin:+8.1f}{b_t1:+9.1f}{b_t2:+9.1f}"
          f"{B_r:8.3f}{B_t:7.3f}")
print("""
判读：T1/T2 掉到真实Δ一带 → 外环衰减=真凶；不掉 → 还有隐藏皮。""")
