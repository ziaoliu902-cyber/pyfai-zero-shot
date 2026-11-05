# -*- coding: utf-8 -*-
"""Week1 前置探测 v2：绕开 pyFAI mask rebin 的奇数维问题（跑完即弃）"""
import time
import numpy as np
import pyFAI
from pyFAI.detectors import detector_factory
from pyFAI.geometry import Geometry

print("pyFAI", pyFAI.version)

# 1) Pilatus 1M：原生 mask 先取，再 binning
det = detector_factory("Pilatus_1M")
mask_native = det.mask.copy()          # (1043, 981) bool
print("native:", det.shape, "mask dead px:", int(mask_native.sum()))
det.binning = (2, 2)                    # 新版属性写法
H, W = det.shape
print("binned:", det.shape, "pixel1/2 [m]:", det.pixel1, det.pixel2)
# 手工构造 binned 死区（含原生 gap 的 bin 视为死）
Hn, Wn = 1042, 980                      # 裁成偶数
mask_binned = mask_native[:Hn, :Wn].reshape(H, 2, W, 2).any(axis=(1, 3))
print("binned dead bins:", int(mask_binned.sum()), "/", H*W)

# 2) q map
g = Geometry(detector=det, dist=1.2, poni1=0.08, poni2=0.09,
             rot1=0.02, rot2=-0.03, rot3=0.05, wavelength=1.5406e-10)
t0 = time.perf_counter()
qmap = g.array_from_unit(shape=det.shape, typ="center", unit="q_A^-1", scale=True)
t1 = time.perf_counter()
print(f"qmap {qmap.shape} dtype={qmap.dtype} NaN={int(np.isnan(qmap).sum())} | {1e3*(t1-t0):.1f} ms")
print("q range: %.4f -> %.3f A^-1" % (np.nanmin(qmap), np.nanmax(qmap)))

# 3) 环半径解析验证（rot=0，束流在探测器中心）
g0 = Geometry(detector=det, dist=1.5, poni1=det.pixel1*H/2,
              poni2=det.pixel2*W/2, rot1=0, rot2=0, rot3=0,
              wavelength=1.5406e-10)
q0 = g0.array_from_unit(shape=det.shape, typ="center", unit="q_A^-1", scale=True)
cy, cx = H/2, W/2
r_px = 60
val_up = q0[int(cy - r_px), int(cx)]
val_down = q0[int(cy + r_px), int(cx)]
two_theta = np.arctan(r_px * det.pixel1 / 1.5)
lam = 1.5406
q_theory = 4*np.pi*np.sin(two_theta/2) / lam
print(f"ring check: q(up)={val_up:.5f} q(down)={val_down:.5f} theory={q_theory:.5f} A^-1")

# 4) 单样本耗时摸底（binned 分辨率，7 峰 Lorentzian + Poisson）
rng = np.random.default_rng(0)
def synth_once():
    g = Geometry(detector=det, dist=10**rng.uniform(-0.6, 0.48),
                 poni1=rng.uniform(0.08, 0.92)*det.pixel1*H,
                 poni2=rng.uniform(0.08, 0.92)*det.pixel2*W,
                 rot1=rng.uniform(-0.07, 0.07), rot2=rng.uniform(-0.07, 0.07),
                 rot3=rng.uniform(-0.17, 0.17), wavelength=1.5406e-10)
    q = g.array_from_unit(shape=det.shape, typ="center", unit="q_A^-1", scale=True)
    q = np.nan_to_num(q, nan=100.0, posinf=100.0)
    img = np.zeros_like(q)
    for _ in range(7):
        q0p = rng.uniform(0.2, 3.5); sig = rng.uniform(0.002, 0.01)
        A = rng.uniform(0.3, 1.0)
        img += A * sig**2 / ((q - q0p)**2 + sig**2)
    img += rng.uniform(0.0, 0.05) * q
    img *= rng.uniform(2e3, 2e4)
    img = rng.poisson(img).astype(np.float32)
    img[mask_binned] = 0
    return img.astype(np.uint16)

t0 = time.perf_counter()
N = 5
for _ in range(N):
    im = synth_once()
t1 = time.perf_counter()
ms = (t1-t0)/N*1e3
print(f"synth once: {ms:.1f} ms -> 50k 单进程 {50_000*ms/1e3/60:.1f} min, 6进程约 {50_000*ms/1e3/60/5.5:.1f} min")
print("img stats: min=%d max=%d mean=%.1f dead=%d" % (im.min(), im.max(), im.mean(), int((im==0).sum())))
