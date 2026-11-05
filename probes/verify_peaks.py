# -*- coding: utf-8 -*-
"""终极验证：同一 Geometry 下用 pyFAI 对合成图做 1D 积分，峰位必须落在表格 q0。

图像的 q-map 由 make_geometry 生成，积分也用同一 Geometry —— 峰位对上即证明
geometry→q-map→合成图像 全链路自洽。跑完即弃。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from copy import deepcopy
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from torchfai.config import SimConfig
from torchfai.simulator import (get_detector, synth_sample, ring_radius_m,
                                phase_qi, make_geometry)
from torchfai.phases import PHASE_NAMES

cfg = SimConfig()
db = get_detector(cfg.detector_name, cfg.binning)
lam_A = cfg.wavelength * 1e10
px = db.pixel
rng = np.random.default_rng(7)

# 与 preview 脚本完全相同的采样序列（同种子同消耗）
picks, seen = [], set()
for _ in range(80):
    s = synth_sample(rng, cfg, db)
    ph = s["phase"]
    if len(picks) < 9 and (ph not in seen or len(seen) == len(PHASE_NAMES)):
        picks.append(s)
        seen.add(ph)
    if len(picks) >= 9:
        break


def find_peaks(q, I, n=8):
    """轻平滑 + 局部极大 + 简单基线扣除，返回 (q, prominence) 列表。"""
    Is = np.convolve(I, np.ones(9) / 9, mode="same")
    base = np.convolve(Is, np.ones(121) / 121, mode="same")
    prom = Is - base
    idx = [i for i in range(6, len(q) - 6)
           if prom[i] == max(prom[max(0, i - 8):i + 9]) and prom[i] > 0.02 * prom.max()]
    idx = sorted(idx, key=lambda i: -prom[i])[:n]
    return sorted([(float(q[i]), float(prom[i])) for i in idx])


print(f"{'sample':>16} {'q0':>8} {'q_meas':>9} {'dq':>8} {'~dr_px':>7}  verdict")
worst = 0.0
for i, s in enumerate(picks):
    g, ph = s["geom"], s["phase"]
    gobj = make_geometry(g, cfg, db)
    ai = AzimuthalIntegrator()
    ai.setPyFAI(**gobj.getPyFAI())
    # detector.mask 在 binning 后尺寸不对（pyFAI 奇数维 rebin 已知问题）——
    # deepcopy 置空它，改用我们手算的 db.dead 作外置 mask
    ai.detector = deepcopy(ai.detector)
    ai.detector.mask = None
    q1d, I1d = ai.integrate1d(s["img"].astype(np.float64), 3000,
                              unit="q_A^-1", method="bbox", mask=db.dead)
    peaks = find_peaks(q1d, I1d)
    qs, Is = phase_qi(ph)
    Hm, Wm = db.extent
    corners = np.array([g["poni1"], Hm - g["poni1"], g["poni2"], Wm - g["poni2"]])
    r_far = float(np.hypot(max(corners[0], corners[1]), max(corners[2], corners[3])))
    bs_r = float(s["aux"][4])
    thr = (bs_r + 3.0) * px if bs_r > 0 else 0.0
    rr = ring_radius_m(qs, g["dist"], lam_A)
    vis = (rr > thr) & (rr < 1.05 * r_far)
    top = sorted([k for k in range(len(qs)) if vis[k]], key=lambda k: -Is[k])[:2]
    for k in top:
        q0 = qs[k]
        if peaks:
            cand = min(peaks, key=lambda t: abs(t[0] - q0))
        else:
            cand = (float("nan"), 0.0)
        dq = cand[0] - q0
        dr = abs(dq) / max(q0, 1e-9) * rr[k] / px
        worst = max(worst, dr)
        tag = "OK" if dr < 2.0 else "MISALIGN"
        print(f"{i:02d}_{ph:>12} {q0:8.4f} {cand[0]:9.4f} {dq:+8.4f} {dr:7.2f}  {tag}")
print(f"--- worst ~dr = {worst:.2f} px (pass < 2 px) ---")
