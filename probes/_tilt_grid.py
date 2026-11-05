# -*- coding: utf-8 -*-
"""_tilt_grid.py：倾角拟合参数网格（帧0000，对 GT rot=(15.82, 9.40)mrad）。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import fabio

from poni_out import fit_tilt
from solver_core import ring_radii, solve, to_canvas
from torchfai.phases import PHASES, phase_qs

AL = os.path.join(ROOT, "data", "real", "alumina")
txt = dict(l.split(":", 1) for l in open(
    os.path.join(AL, "al2o3_00_max_51_frames.poni"))
    if ":" in l and not l.startswith("#"))
lam = float(txt["Wavelength"])
px1, px2 = float(txt["PixelSize1"]), float(txt["PixelSize2"])
gr = (float(txt["Rot1"]), float(txt["Rot2"]))

frames = [f"al2o3_{i:04d}.edf" for i in (0, 10, 30)]
for fr in frames:
    img = fabio.open(os.path.join(AL, fr)).data.astype(np.float64)
    canvas, s, cx0, cy0 = to_canvas(img)
    Q = phase_qs("Al2O3")
    peaks, prof, bp, gd = ring_radii(canvas, cx0, cy0)
    sol = solve(peaks, cx0, cy0, 1.0 / ((px1 + px2) / 2 * s), Q, lam,
                (px1 + px2) / 2 * s, I_rel=PHASES["Al2O3"][1],
                prof_ctx=(prof, bp, gd), canvas=canvas)
    if not sol or sol[3] < 5:
        print(f"{fr}: 圆解失败")
        continue
    cx, cy, D, m, rms = sol[:5]
    print(f"\n== {fr} 圆解D={D*(px1+px2)/2*s*1000:.2f}mm m={m} rms={rms:.2f}px "
          f"GTrot=({gr[0]*1e3:+.2f},{gr[1]*1e3:+.2f})mrad")
    for nsec, win, mr, bp in [(12, 6, 0.12, 1.25), (18, 6, 0.12, 1.5),
                              (24, 6, 0.12, 1.5), (18, 7, 0.10, 1.4),
                              (24, 7, 0.10, 1.5), (36, 7, 0.10, 1.5),
                              (24, 8, 0.08, 1.5)]:
        tf = fit_tilt(canvas, cx, cy, D, Q, lam, px1 * s, px2 * s,
                      nsec=nsec, win=win, mass_rel=mr, bump=bp)
        if not tf:
            print(f"  nsec={nsec} win={win} mr={mr} bp={bp}: 控制点不足")
            continue
        print(f"  nsec={nsec:2d} win={win} mr={mr} bp={bp}: "
              f"{tf['npts']:3d}pt/{tf['nrings']:2d}环 "
              f"rms3={tf['rms3']*1e3:5.2f}→rms5={tf['rms5']*1e3:5.2f}mrad "
              f"改善{tf['imp']*100:4.0f}% "
              f"rot=({tf['rot1']*1e3:+6.1f},{tf['rot2']*1e3:+6.1f}) "
              f"Δ=({(tf['rot1']-gr[0])*1e3:+5.1f},{(tf['rot2']-gr[1])*1e3:+5.1f}) "
              f"tilt={tf['tilt_deg']:.2f}°")
