# -*- coding: utf-8 -*-
"""_b11c_centers.py:B11c 定心算法偏斜稳健性判决(2026-11-07)。

B11b 教训:径向峰形偏斜(物理源:色散/样品厚度吸收/织构方位调制)
等量偏移一切"找峰位"法(1D 谱、扇区控制点、5 参拟合)——几何层无解。
出路在**定心算子**:对比四种定位在 0.3px 注入偏斜下的 Δr:

  P 抛物线众数(近生产)   —— 对称高斯最优,偏斜敏感
  C 质心(窗口内)        —— 偏斜全吸收(最差预期)
  M 半高中点(上升/下降沿中点) —— 对"单侧宽化"一阶免疫
  S 分裂高斯拟合(center 参数) —— 建模偏斜,预期最优

判决:M/S 把 0.3px 偏斜压到 <0.1px → 生产定心可换。
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from torchfai.phases import phase_qs
from _b11_budget import synth, LAM, D_MM, PX_UM, Q

PX = PX_UM * 1e-6


def radial(img):
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W].astype(float)
    cx, cy = W * 0.495, H * 0.502
    r = np.hypot(x - cx, y - cy)
    nb = np.arange(0, r.max() + 2, 1.0)
    prof, _ = np.histogram(r, bins=nb, weights=img)
    cnt, _ = np.histogram(r, bins=nb)
    return nb[:-1], prof / np.maximum(cnt, 1)


def centers(rb, I, r0, win=8.0):
    m = (rb > r0 - win) & (rb < r0 + win)
    x, y = rb[m], I[m]
    if len(x) < 6:
        return {}
    out = {}
    # P 抛物线(三点/加权众数)
    i = int(np.argmax(y))
    if 0 < i < len(x) - 1:
        dy = (y[i - 1] - 2 * y[i] + y[i + 1])
        out["P"] = x[i] + 0.5 * (y[i - 1] - y[i + 1]) / dy \
            if dy != 0 else x[i]
    # C 质心(去本底)
    b = np.percentile(y, 10)
    w = np.clip(y - b, 0, None)
    out["C"] = float((x * w).sum() / w.sum())
    # M 半高中点
    half = (y.max() - b) / 2 + b
    li = np.where((y[:-1] < half) & (y[1:] >= half))[0]
    ri = np.where((y[:-1] >= half) & (y[1:] < half))[0]
    if len(li) and len(ri):
        lo = np.interp(half, [y[li[-1]], y[li[-1] + 1]],
                       [x[li[-1]], x[li[-1] + 1]])
        hi = np.interp(half, [y[ri[0] + 1], y[ri[0]]],
                       [x[ri[0] + 1], x[ri[0]]])
        out["M"] = 0.5 * (lo + hi)
    # S 分裂高斯
    try:
        p0 = [x[np.argmax(y)], y.max(), 2.0, 2.0, b]

        def resid(p):
            c, a, sL, sR, bb = p
            s = np.where(x < c, sL, sR)
            return a * np.exp(-0.5 * ((x - c) / s) ** 2) + bb - y
        fit = least_squares(resid, p0, loss="soft_l1", f_scale=2.0)
        out["S"] = float(fit.x[0])
    except Exception:
        pass
    return out


tt = 2 * np.arcsin(np.clip(Q * LAM / (4 * np.pi), 0, 0.999))
R0 = (D_MM * 1e-3 / PX) * np.tan(tt)

img0 = synth(skew_px=0.0)
img1 = synth(skew_px=0.3)
rb, I0 = radial(img0)
_, I1 = radial(img1)

acc = {}
for r0 in R0:
    if r0 < 40 or r0 > 1200:
        continue
    c0, c1 = centers(rb, I0, r0), centers(rb, I1, r0)
    for k in c0:
        if k in c1:
            acc.setdefault(k, []).append(c1[k] - c0[k])

print("=" * 62)
print("B11c 定心算子对 0.3px 径向偏斜的响应(Δr,px)")
print("=" * 62)
for k in ("P", "C", "M", "S"):
    v = np.array(acc.get(k, []))
    if len(v):
        print(f"  {k:2} n={len(v)}  中位 Δr={np.median(v):+.3f}px  "
              f"MAD={np.median(np.abs(v - np.median(v))):.3f}px")
print("\n判决:|中位|<0.1px 的算子对偏斜稳健(0.3px→<0.1px 即 3× 压缩)")
