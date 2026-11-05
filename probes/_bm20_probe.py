# -*- coding: utf-8 -*-
r"""_bm20_probe.py — BM20 4 真值帧结构侦察：
1) 完整读 4 poni；
2) R 增量 vs delta 线性度（log(ΔR) 回归 → 轴 + 角速率 k，期望 k≈1 rad/rad）；
3) 每分量 (1,cos,sin) LOO 内插误差（病态检查）。
"""
import os
import re

import numpy as np

BASE = r"C:\Users\liu\Desktop\衍射测试图\ext1\LaB6_gonio_BM20"
TAGS = ["31.5", "33.5", "35.0", "35.5"]

ponis = {}
for t in TAGS:
    txt = open(os.path.join(BASE, "del_%s_0001p.poni" % t),
               encoding="utf-8", errors="replace").read()
    d = dict(re.findall(r"^(Distance|Poni1|Poni2|Rot1|Rot2|Rot3|Wavelength):\s*(\S+)",
                        txt, re.M))
    ponis[t] = {k: float(v) for k, v in d.items()}
    p = ponis[t]
    print("del_%s: D=%.4f poni=(%.5f,%.5f) rot=(%.6f,%.6f,%.6f) wl=%.4e"
          % (t, p["Distance"], p["Poni1"], p["Poni2"],
             p["Rot1"], p["Rot2"], p["Rot3"], p["Wavelength"]))

def rmat(r1, r2, r3):
    sx, cx = np.sin(r1), np.cos(r1)
    sy, cy = np.sin(r2), np.cos(r2)
    sz, cz = np.sin(r3), np.cos(r3)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def log_so3(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-12:
        return np.zeros(3), 0.0
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * th / (2 * np.sin(th)), th

# --- 1. R-increment linearity vs delta (deg) ---
g0 = TAGS[0]
R0 = rmat(*[ponis[g0][k] for k in ("Rot1", "Rot2", "Rot3")])
print("\n== R increments vs del_%s ==" % g0)
xs, ys = [], []
for t in TAGS[1:]:
    Ri = rmat(*[ponis[t][k] for k in ("Rot1", "Rot2", "Rot3")])
    w, th = log_so3(Ri @ R0.T)
    dd = float(t) - float(g0)
    print("d=%.1fdeg: |w|=%.4fdeg axis=%s" % (dd, np.degrees(th), np.round(w / np.linalg.norm(w), 3)))
    xs.append(np.radians(dd))
    ys.append(w)
xs, ys = np.array(xs), np.array(ys)
A = np.stack([xs, np.ones_like(xs)], 1)
slope_axis, _ = np.linalg.lstsq(A, ys, rcond=None)[:2]
k = np.linalg.norm(slope_axis[0])
axis = slope_axis[0] / k
pred = A @ np.stack([slope_axis[0], slope_axis[1]])
res = np.degrees(np.linalg.norm(ys - pred, axis=1))
print("k (rad/rad) = %.4f  axis=%s  resid(deg)=%s"
      % (k, np.round(axis, 3), np.round(res, 3)))

# --- 2. per-component (1,cos,sin) LOO on raw poni values ---
print("\n== LOO (1,cos,sin) per component ==")
g = np.radians(np.array([float(t) for t in TAGS]))
comp_names = ["Distance", "Poni1", "Poni2", "Rot1", "Rot2", "Rot3"]
A_full = np.stack([np.ones_like(g), np.cos(g), np.sin(g)], 1)
print("cond(A_full) = %.1f" % np.linalg.cond(A_full))
for cn in comp_names:
    y = np.array([ponis[t][cn] for t in TAGS])
    errs = []
    for i in range(4):
        m = np.ones(4, bool); m[i] = False
        c, *_ = np.linalg.lstsq(A_full[m], y[m], rcond=None)
        errs.append(y[i] - A_full[i] @ c)
    errs = np.array(errs)
    scale = max(abs(y).max(), 1e-12)
    print("%-9s LOO err: %s  (rel %s)"
          % (cn, np.array2string(errs, precision=6),
             np.array2string(errs / scale, precision=2)))
