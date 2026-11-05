# -*- coding: utf-8 -*-
r"""_bm20_gen.py — 生成 BM20 121 帧参考 poni 库（置信标注）。

模型: rot2(delta)=rot2_31.5+rad(delta-31.5)+rad(0.15 中段系统偏置校正仅在 mid 段)
     D=0.8056, poni=(0.00563, 0.04612), rot1=rot3=0, wl=7.749012e-11
置信: mid 11<=delta<=35 （对比度检验成立, rot2 精度~0.2deg） -> conf=mid
      其余（环出画布/弱环）-> conf=low
输出: ext1/LaB6_gonio_BM20/gen_poni/del_XX.X_0001p.poni + gen_manifest.md
"""
import glob
import os
import re
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

BASE = r"C:\Users\liu\Desktop\衍射测试图\ext1\LaB6_gonio_BM20"
OUT = os.path.join(BASE, "gen_poni")
os.makedirs(OUT, exist_ok=True)

D, P1, P2, R20 = 0.8056, 0.00563, 0.04612, 0.560707
WL = 7.749012e-11
PX = 172e-6
BIAS_MID = np.radians(0.15)   # mid-range consistent detuning-peak offset

TPL = ("# Generated reference geometry (clean rot2-linear model, k=1)\n"
       "# anchored on official del_31.5 + del_35.0 ponis (SO(3) k=1.003)\n"
       "# conf={conf} (mid: 11<=delta<=35 contrast-verified ~0.2deg; low: rings out-of-view/weak)\n"
       "poni_version: 2\n"
       "Detector: Pilatus100k\n"
       "Detector_config: {{\"pixel1\": {px}, \"pixel2\": {px}}}\n"
       "Distance: {d}\n"
       "Poni1: {p1}\n"
       "Poni2: {p2}\n"
       "Rot1: 0.0\n"
       "Rot2: {r2}\n"
       "Rot3: 0.0\n"
       "Wavelength: {wl}\n")

files = sorted(glob.glob(os.path.join(BASE, "del_*_0001p.cbf")))
n_mid = n_low = 0
rows = []
for f in files:
    tag = float(re.search(r"del_([\d.]+)_0001p", os.path.basename(f)).group(1))
    mid = 11.0 <= tag <= 35.0
    r2 = R20 + np.radians(tag - 31.5) + (BIAS_MID if mid else 0.0)
    conf = "mid" if mid else "low"
    name = "del_%04.1f_0001p.poni" % tag
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(TPL.format(conf=conf, px=PX, d=D, p1=P1, p2=P2, r2=r2, wl=WL))
    n_mid += mid
    n_low += (not mid)
    rows.append((tag, conf, np.degrees(r2)))

print("generated %d ponis (mid=%d low=%d) -> %s" % (len(rows), n_mid, n_low, OUT))

md = ["# BM20 gen_poni 参考几何库（模型生成, 非官方 GT）", "",
      "模型: rot2(δ)=rot2_31.5 + rad(δ-31.5) (+0.15° mid 校正), D=0.8056, poni=(0.00563,0.04612), wl=7.749012e-11",
      "锚定: 官方 del_31.5+del_35.0（SO(3) 增量 3.5124° vs 真值 3.5°, 轴=纯 rot2）；官方 33.5(半步异常)/35.5(rot1 异常)剔除",
      "检验: docs/bm20_model_check.md（全帧环带对比度）+ _bm20_sweep2（中段失谐峰位 +0.10~+0.20° 一致）", "",
      "| delta | conf | rot2(deg) |", "|---|---|---|"]
for t, c, r in rows:
    md.append("| %.1f | %s | %.4f |" % (t, c, r))
md += ["", "计数: mid(11-35°, 对比度验证, ~0.2°) %d 帧; low(环出画布/弱环) %d 帧" % (n_mid, n_low),
       "用途: E2 扩样本弱参考; 亚像素级对打不适用（官方 poni 亦仅 4 帧且 2 帧异常）。"]
with open(os.path.join(OUT, "gen_manifest.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")
print("manifest -> gen_poni/gen_manifest.md")
