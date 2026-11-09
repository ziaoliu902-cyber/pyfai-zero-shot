# -*- coding: utf-8 -*-
"""_b11e_partial.py:B11e 部分环病态复现(2026-11-07,机理验证)。

B11-G:同心遮挡(中心在场内)D 只 −0.07mm,1D 无偏。
Fairchild 病态 = 束心出画布 + rot1=−29°:复现并演示 1D 谱为何死。
模型:中心 x 出画布 300px;倾角一阶 2θ→2θ+rot·cos(φ);遮上侧 55%。
看:① 1D 径向谱(solver_core.radial_peaks,参考=画布几何)还能找到
      几个真环、半径偏多少;② 同图"真中心"参考下的谱(上界对照)。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc
from torchfai.phases import phase_qs

rng = np.random.default_rng(11)
PX = 172e-6
LAM = 0.972386
D_MM = 200.19
H, W = 2527, 2463
Q = phase_qs("CeO2")[:9]
tt = 2 * np.arcsin(np.clip(Q * LAM / (4 * np.pi), 0, 0.999))

cx = -300.0                      # 束心出画布 300px(Fairchild 型)
cy = H * 0.50
rot1 = np.deg2rad(-29.0)         # 面内大倾角

y, x = np.mgrid[0:H, 0:W].astype(float)
r = np.hypot(x - cx, y - cy)
phi = np.arctan2(y - cy, x - cx)
img = np.zeros((H, W))
for tq in tt:
    dtt = rot1 * np.cos(phi)               # 一阶倾角
    rr = D_MM * 1e-3 / PX * np.tan(tq + dtt)
    img += 3000.0 * np.exp(-0.5 * ((r - rr) / 1.5) ** 2)
img[phi > np.pi * 0.55] = 0.0              # 遮上侧 ~55% 方位(部分环)
img = rng.poisson(img).astype(float)

R_true = D_MM * 1e-3 / PX * np.tan(tt)

print("=" * 70)
print("B11e 部分环病态:束心出画布 300px + rot1=−29° + 遮 55% 方位")
print("=" * 70)
# ① 生产口径:参考中心 = 画布几何(solver 盲解时对出画布中心一无所知)
canvas, s, _, _ = sc.to_canvas(img)
for tag, ccx, ccy in (("画布中心参考(盲解视角)", canvas.shape[1] * 0.5,
                       canvas.shape[0] * 0.5),
                      ("真中心参考(上界对照)", cx * s, cy * s)):
    pk, prof, bp, good = sc.ring_radii(canvas, ccx, ccy)
    hit = 0
    bias = []
    for rj in R_true * s:
        near = [p for p in pk if abs(p[0] - rj) < 0.05 * rj]
        if near:
            hit += 1
            bias.append(near[0][0] - rj)
    print(f"\n[{tag}] 找到峰 {len(pk)} 个,命中真环 {hit}/9")
    if bias:
        print(f"  命中环半径偏差:中位 {np.median(bias):+.1f}px "
              f"(max {np.max(np.abs(bias)):.1f}px)")
    print(f"  峰半径(px@画布):{[round(p[0]) for p in pk[:12]]}")
print("\n真环半径(px@原生):", [round(v) for v in R_true])
print("\n机理:部分弧+椭圆(倾角)在 1D 方位平均里把 r(φ) 分布糊成宽带,")
print("峰位 ≠ 任何几何半径;中心出画布让'以画布中心为参考'的谱雪上加霜。")
print("结论:部分环必须 2D 弧段(斑点半径-方位控制点)+ 多环共享几何。")
