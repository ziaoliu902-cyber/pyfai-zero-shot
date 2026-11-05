# -*- coding: utf-8 -*-
"""折叠对决：环心到底在 GT poni (194,240) 还是直测 (263.6,179)？

同图按两套几何做 2θ 径向积分，比 1~5 号环区的峰高/半高宽。
峰锐者胜——环真实的中心使所有环同时聚焦。
"""
import os
import sys

import numpy as np
import fabio
import pyFAI

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
REAL = os.path.join(ROOT, "data", "real")
PX = 172e-6

img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
p = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))

# 几何 A：参考 poni 原封不动
geoA = dict(dist=p.dist, poni1=p.poni1, poni2=p.poni2,
            rot1=p.rot1, rot2=p.rot2, rot3=p.rot3,
            detector="Pilatus_1M", wavelength=1e-10)
# 几何 B：直测环心 + 同梯子 dist（自洽解 D=1.6263）
geoB = dict(dist=1.6263, poni1=263.6 * PX, poni2=179.0 * PX,
            rot1=0.0, rot2=0.0, rot3=0.0,
            detector="Pilatus_1M", wavelength=1e-10)

# AgBh Bragg 2θ（λ=1e-10 m, d0=58.38 Å）：谐波梯
q0 = 2 * np.pi / 58.38e-10
tth_b = [np.degrees(np.arcsin(q0 * n * 1e-10 / (4 * np.pi))) * 2
         for n in range(1, 6)]


def fold_peaks(geo, tag):
    itg = pyFAI.AzimuthalIntegrator(**{k: v for k, v in geo.items()})
    q, I = itg.integrate1d(img, 4000, unit="2th_deg",
                           radial_range=(0.5, 6.0))
    print(f"\n== 几何 {tag} ==")
    tot = 0.0
    for i, t0 in enumerate(tth_b, 1):
        w = (q > t0 - 0.10) & (q < t0 + 0.16)
        if w.sum() < 20:
            continue
        qq, ii = q[w], I[w]
        k = int(np.argmax(ii))
        # 半高宽
        half = (ii[k] + np.median(ii)) / 2
        above = ii > half
        l = k
        while l > 0 and above[l]:
            l -= 1
        r_ = k
        while r_ < len(ii) - 1 and above[r_]:
            r_ += 1
        fwhm = qq[r_] - qq[l]
        snr = (ii[k] - np.median(ii)) / (np.std(np.sort(ii)[: len(ii) // 3]) + 1e-9)
        tot += snr
        print(f"  环{i}  Bragg={t0:.3f}°  峰位={qq[k]:.3f}°  "
              f"FWHM={fwhm:.3f}°  SNR={snr:6.1f}")
    print(f"  ΣSNR = {tot:.1f}")
    return tot


sA = fold_peaks(geoA, "A·参考 poni (194,240)")
sB = fold_peaks(geoB, "B·直测环心 (263.6,179)")
print(f"\n判决：{'A 参考束心' if sA > sB else 'B 直测束心'} 胜 "
      f"(ΣSNR {sA:.0f} vs {sB:.0f}, 比值 {max(sA, sB)/max(min(sA, sB), 1e-9):.2f}×)")
