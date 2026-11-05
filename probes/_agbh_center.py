# -*- coding: utf-8 -*-
"""AgBh 环心仲裁：图像直测环心 vs GT poni vs NN poni。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
import fabio
import pyFAI

from real_probe import to_nn_input_phys
from real_refine_v6 import REAL, ROOT, load_px_nn, match_gain, nn_coarse

m, dev, consts = load_px_nn(os.path.join(ROOT, "runs/px_v2/best.pt"))
img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
p = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))
px = 344e-6                                  # 画布有效像素

c = match_gain(to_nn_input_phys(img)) * 0.5
p_nn, _ = nn_coarse(m, dev, consts, c, 1.0, 1.0)

# 图像直测：高亮像素（环）坐标中位 + 四象限对称质心
v = c.astype(np.float64)
thr = np.percentile(v, 99.99)
ys, xs = np.where(v >= thr)
gt_y, gt_x = p.poni1 / px, p.poni2 / px
print(f"画布 {v.shape}，GT poni=(y {gt_y:.0f}, x {gt_x:.0f})px")
print(f"NN  poni=(y {p_nn[1]/px:.0f}, x {p_nn[2]/px:.0f})px "
      f"(Δy {(p_nn[1]-p.poni1)/px:+.0f} Δx {(p_nn[2]-p.poni2)/px:+.0f})")
print(f"高亮像素({len(ys)}) 中位中心=(y {np.median(ys):.0f}, "
      f"x {np.median(xs):.0f})px")
print(f"高亮像素     质心=(y {ys.mean():.0f}, x {xs.mean():.0f})px")

# 径向对称中心粗估：对 Top0.1% 像素求使径向方差最小的中心（网格搜索）
best = (None, np.inf)
ys_f, xs_f = ys.astype(float), xs.astype(float)
w = v[ys, xs]
for cy in range(100, 460, 20):
    for cx in range(80, 440, 20):
        r = np.hypot(ys_f - cy, xs_f - cx)
        s = (w * (r - (w * r).sum() / w.sum()) ** 2).sum() / w.sum()
        if s < best[1]:
            best = ((cy, cx), s)
print(f"径向对称中心（粗网格）= (y {best[0][0]}, x {best[0][1]})px")

# GT 帧级复核：poni 文件里的 dist 与像素尺寸
print(f"\nGT poni 全参数: dist={p.dist:.4f} poni=({p.poni1:.4f},"
      f"{p.poni2:.4f}) rot=({p.rot1:.5f},{p.rot2:.5f},{p.rot3:.5f})")
print(f"detector: {p.detector.name if p.detector else '?'} "
      f"pixel={p.pixel1*1e6:.0f}x{p.pixel2*1e6:.0f}µm "
      f"spline={p.spline_file}")
