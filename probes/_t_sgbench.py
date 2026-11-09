# -*- coding: utf-8 -*-
"""_t_sgbench.py:splitG 单次计时+调用量估算(2026-11-08)。"""
import sys
import time

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os = __import__("os")
os.chdir(ROOT)

import solver_core as sc

# 真实径向谱(Pilatus2M)
img = fabio.open(ROOT + r"\data\real\silx_zoo\Pilatus2M.cbf").data
img = np.where(np.isfinite(img), img, 0.0).astype(np.float64)
H, W = img.shape
yy, xx = np.mgrid[0:H:4, 0:W:4]          # 1/16 画布近似
prof_axis = np.sqrt((xx - W / 8) ** 2 + (yy - H / 8) ** 2).ravel() * 4
nb = 1024
idx = np.clip((prof_axis / (max(prof_axis) / nb)).astype(int), 0, nb - 1)
prof = np.zeros(nb)
np.add.at(prof, idx, img[::4, ::4].ravel())

# 找几个峰位
pk = np.argsort(prof)[::-1][:200]
pk = sorted(set(int(p) for p in pk if 2 < p < nb - 3))
print(f"谱 {nb} bin,候峰 {len(pk)} 个")

from scipy.optimize import least_squares

t0 = time.time()
n_ok = 0
for p in pk[:50]:
    r = sc._split_gauss_center(prof, p, float(np.median(prof)))
    n_ok += r is not None
dt = (time.time() - t0) / 50
print(f"单次 _split_gauss_center:{dt*1e3:.2f}ms(成功 {n_ok}/50)")

# 内层剖析:一次 least_squares 的 nfev
p = pk[0]
lo, hi = max(0, p - 6), min(nb, p + 7)
x = np.arange(lo, hi, dtype=float)
y = prof[lo:hi]
p0 = [float(p), float(prof[p]), 2.0, 2.0, float(np.median(prof))]
calls = [0]

def resid(q):
    calls[0] += 1
    c, a, sL, sR, b = q
    s = np.where(x < c, np.abs(sL), np.abs(sR)) + 1e-6
    return a * np.exp(-0.5 * ((x - c) / s) ** 2) + b - y

t0 = time.time()
fit = least_squares(resid, p0, loss="soft_l1", f_scale=2.0,
                    bounds=([p - 1.5, 0, 0.3, 0.3, -np.inf],
                            [p + 1.5, np.inf, 20, 20, np.inf]))
print(f"剖析:nfev={calls[0]} 耗时={(time.time()-t0)*1e3:.1f}ms "
      f"status={fit.status} x={fit.x[:3].round(2)}")
