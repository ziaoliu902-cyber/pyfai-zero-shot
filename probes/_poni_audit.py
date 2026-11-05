# -*- coding: utf-8 -*-
"""参考 poni 自洽性的独立审计（不依赖我们的 refiner/模型）。

原理：若参考标定 (D=1583.2mm, λ=1.00Å) 自洽，则用该几何把图像
积成 I(2θ) 后，AgBh 的环必须精确落在布拉格角
2θ_n = 2·asin(q_n·λ/4π) 上。环若系统性偏移 δ，则证明
(D, λ, 像素) 组合与图像不自洽，且偏移量直接给出乘积误差。

用 pyFAI 本尊完成（参考文件的生成工具链），排除我们代码的嫌疑。
AgBh q 值双轨：文献 d=58.38Å(NIST) 与我们相位表，交叉对照。
"""
import os
import sys

import numpy as np
import fabio
import pyFAI
from scipy.signal import find_peaks

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
REAL = os.path.join(ROOT, "data", "real")

img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
ai = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))
print(f"参考几何: D={ai.dist*1000:.2f}mm  λ={ai.wavelength*1e10:.4f}Å")
print(f"探测器: {ai.detector.name}  {img.shape}")

q_i, I = ai.integrate1d(img, 4000, unit="2th_deg", radial_range=(0.7, 6.0),
                        method="bbox", correctSolidAngle=False)
I = np.asarray(I, float)
pk, _ = find_peaks(I, prominence=np.max(I) * 0.01)
obs = q_i[pk]

for tag, q0 in (("AgBh 文献 d=58.38Å (NIST)", 2 * np.pi / 58.38),
                ("AgBh 相位表 q0=0.108", 0.108)):
    print(f"\n== {tag} ==")
    print(f"{'环':>3s} {'布拉格2θ°':>10s} {'实测2θ°':>9s} {'偏差%':>7s}")
    ds = []
    for n in range(1, 6):
        t = np.degrees(2 * np.arcsin(n * q0 * (ai.wavelength * 1e10)
                                       / (4 * np.pi)))
        m = obs[np.argmin(np.abs(obs - t))]
        d = (m - t) / t * 100
        ds.append(d)
        print(f"{n:>3d} {t:10.4f} {m:9.4f} {d:+7.2f}")
    dm = float(np.mean(ds))
    print(f"平均 {dm:+.2f}% → 自洽需 D×λ 上调 {dm:+.2f}%：")
    print(f"   D={ai.dist*1000*(1+dm/100):.1f}mm（λ=1.00 不变）"
          f" 或 λ={ai.wavelength*1e10*(1+dm/100):.4f}Å（D 不变）")
