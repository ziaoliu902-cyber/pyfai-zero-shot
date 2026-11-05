# -*- coding: utf-8 -*-
"""试验：Hough/径向对称梯度投票定心（部分弧也能用），对象 Pilatus1M / mar3450 / Eiger9M。

思路：环边缘像素的亮度梯度方向沿半径 → 沿 ±梯度方向在 3D (x,y,r) 空间投票，
累加后取峰值。部分圆弧（甚至 60° 弧）也能投出清晰中心。
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import pyFAI.azimuthalIntegrator
import zoo_run as ZR


def radial_symmetry_center(canvas, rmin=6, rmax=None, edge_frac=0.10,
                           smooth=1.0, verbose=False):
    from scipy.ndimage import sobel, gaussian_filter
    img = gaussian_filter(np.clip(canvas, 0, None), smooth)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    mag = np.hypot(gx, gy)
    H, W = img.shape
    if rmax is None:
        rmax = max(H, W) * 0.5
    thr = np.quantile(mag, 1 - edge_frac)
    m = mag > max(thr, 1e-9)
    ys, xs = np.nonzero(m)
    gxs, gys = gx[ys, xs], gy[ys, xs]
    n = np.hypot(gxs, gys) + 1e-12
    ux, uy = gxs / n, gys / n
    acc = np.zeros((H, W))
    rr = np.arange(rmin, rmax, 2.0)
    for r in rr:
        cx = np.round(xs - r * ux).astype(int)
        cy = np.round(ys - r * uy).astype(int)
        ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        np.add.at(acc, (cy[ok], cx[ok]), 1.0)
        # 反向也投（梯度方向可能朝内）
        cx2 = np.round(xs + r * ux).astype(int)
        cy2 = np.round(ys + r * uy).astype(int)
        ok2 = (cx2 >= 0) & (cx2 < W) & (cy2 >= 0) & (cy2 < H)
        np.add.at(acc, (cy2[ok2], cx2[ok2]), 1.0)
    from scipy.ndimage import maximum_filter
    acc = gaussian_filter(acc, 1.5)
    peak = np.unravel_index(np.argmax(acc), acc.shape)
    if verbose:
        print("  grad pixels", int(m.sum()), "vote peak", peak,
              "score", acc[peak])
    return float(peak[1]), float(peak[0]), acc


if __name__ == "__main__":
    items = {
        "Pilatus1M": ("Pilatus1M.edf", "Pilatus1M.poni"),
        "Eiger9M": ("Eiger9M.h5", "Eiger9M.poni"),
        "mar3450": ("LaB6_260210.mar3450", "Mar3450.poni"),
    }
    for tag, (img_f, poni_f) in items.items():
        img = ZR.load_img(os.path.join(ZOO, img_f))
        canvas, s, _, _ = ZR.to_canvas(img)
        ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
        try:
            f2d = ai.getFit2D()
            gt = (f2d["centerX"] / s, f2d["centerY"] / s)
        except Exception:
            gt = (ai.poni2 / ai.detector.pixel2 / s,
                  ai.poni1 / ai.detector.pixel1 / s)
        cx, cy, acc = radial_symmetry_center(canvas)
        print(f"{tag}: canvas={canvas.shape} RS_center=({cx:.1f},{cy:.1f}) "
              f"GT=({gt[0]:.1f},{gt[1]:.1f}) "
              f"d={np.hypot(cx-gt[0], cy-gt[1]):.1f}px")
