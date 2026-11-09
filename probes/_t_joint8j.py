# -*- coding: utf-8 -*-
"""_t_joint8j.py:52 帧 mean 逐字节复刻 phase_fit B 段(2026-11-08)。"""
import glob
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc
from poni_out import fit_tilt, make_pyfai_fwd
from torchfai.phases import phase_qs

txt = dict(l.split(":", 1) for l in open(
    glob.glob(r"data\real\alumina\*.poni")[0])
    if ":" in l and not l.startswith("#"))
lam = float(txt["Wavelength"])
px1, px2 = float(txt["PixelSize1"]), float(txt["PixelSize2"])
gt = float(txt["Distance"])
D_IR = sorted(glob.glob(r"data\real\alumina\*.edf"))
print(f"复刻 phase_fit B 段:{len(D_IR)} 帧,px=({px1*1e6:.3f},"
      f"{px2*1e6:.3f})µm λ={lam*1e10:.7f}")

# 与 phase_fit 逐字节一致:raw 平均,不清洗
imgm = np.mean([fabio.open(f).data for f in D_IR], 0).astype(np.float64)
res = sc.solve_image(imgm, (px1 + px2) / 2, lam=lam, phases=["Al2O3"],
                     solve_kw=dict(cascade=True, auto_center=False,
                                   refine_2d=False))
assert res, "圆解失败"
canvas, s = res["canvas"], res["s"]
sol = res["sol"]
print(f"圆解:D={sol[2]*(px1+px2)/2*s*1000:.3f}mm (GT {gt*1000:.3f}, "
      f"Δ{sol[2]*(px1+px2)/2*s*1000-gt*1000:+.3f}) m={sol[3]} "
      f"rms={sol[4]:.2f}px s={s}")
fwd = make_pyfai_fwd(px1, px2, os.path.join(
    ROOT, "data", "real", "alumina", "distorsion_2x2.spline"))
tf = fit_tilt(canvas, sol[0], sol[1], sol[2], phase_qs("Al2O3"), lam,
              px1 * s, px2 * s, sfull=s, fwd=fwd, nsec=36, mass_rel=0.10,
              bump=1.5)
if tf is None:
    print("fit_tilt: None ✗")
    sys.exit(1)
print(f"fit_tilt(spline):D={tf['dist']*1e3:.4f}mm "
      f"(GT {gt*1e3:.3f}, Δ{tf['dist']*1e3-gt*1e3:+.4f}) "
      f"rot=({np.degrees(tf['rot1']):+.4f},{np.degrees(tf['rot2']):+.4f})°"
      f" rms5={tf['rms5']*1e3:.2f}mrad npts={tf['npts']}")
print("目标复现:alumina_mean.poni D=116.7256(Δ−0.134)")
