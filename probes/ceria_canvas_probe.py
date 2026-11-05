# -*- coding: utf-8 -*-
"""ceria_canvas_probe.py：ceria 改用 to_canvas 朴素画布 + 经典解 + 倾角。

验证 B'（P4 消融 0.2/0.3mm）搬到生产画布后，D 与倾角是否同时不劣于
现行 wide+NN 路径（4.1/8.0mm）。
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

from poni_out import fit_tilt, make_pyfai_fwd
from solver_core import ring_radii, solve, to_canvas
from torchfai.phases import phase_qs

LAM = 0.972386
Q = phase_qs("CeO2")[:8]

print("== ceria：to_canvas 经典路径 + 倾角 ==")
print(f"{'帧':>5}{'GT':>8}{'Δmm':>8}{'环':>3}{'rms':>6}"
      f"{'rot1Δ':>8}{'rot2Δ':>8}{'DponiΔ':>8}{'pt/环':>8}")
es, r1s, r2s, dps = [], [], [], []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    canvas, s, cx0, cy0 = to_canvas(img)
    px = 172e-6 * s
    peaks, prof, bin_px, good = ring_radii(canvas, cx0, cy0)
    sol = solve(peaks, cx0, cy0, 1.0 / px, Q, LAM, px,
                d_range=(0.05, 3.0), rms_floor=1.8, canvas=canvas,
                prof_ctx=(prof, bin_px, good))
    if sol is None:
        print(f"{d:>5}{gt:8.1f}{'失败':>8}")
        es.append(999.0)
        continue
    cx, cy, D, m, rms = sol[:5]
    dmm = D * px * 1000
    e = dmm - gt
    es.append(abs(e))
    fwd = make_pyfai_fwd(172e-6, 172e-6)   # sfull>1: 全分辨率索引→原生像素
    tf = fit_tilt(canvas, cx, cy, D, Q, LAM, px, px, sfull=s, fwd=fwd,
                  nsec=36, mass_rel=0.10, bump=1.5)
    if tf:
        r1 = (tf["rot1"] - p.rot1) * 1e3
        r2 = (tf["rot2"] - p.rot2) * 1e3
        dp = (tf["dist"] * 1000 - gt)
        r1s.append(r1)
        r2s.append(r2)
        dps.append(dp)
        print(f"{d:>5}{gt:8.1f}{e:+8.1f}{m:>3}{rms:6.2f}"
              f"{r1:+8.1f}{r2:+8.1f}{dp:+8.1f}"
              f"{tf['npts']:>5}/{tf['nrings']}")
    else:
        print(f"{d:>5}{gt:8.1f}{e:+8.1f}{m:>3}{rms:6.2f}  倾角失败")

print(f"\nD: 平均|Δ|={np.mean(es):.2f}mm 最差{max(es):.2f}mm")
if r1s:
    print(f"倾角: rot1Δ均值{np.mean(r1s):+.1f}mrad "
          f"(范围{min(r1s):+.1f}~{max(r1s):+.1f})  "
          f"rot2Δ均值{np.mean(r2s):+.1f}mrad "
          f"(范围{min(r2s):+.1f}~{max(r2s):+.1f})  "
          f"poni DΔ均值{np.mean(dps):+.2f}mm 最差{max(np.abs(dps)):.2f}mm")

