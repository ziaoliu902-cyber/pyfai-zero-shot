# -*- coding: utf-8 -*-
"""把 zoo 失败项画成 PNG：原始图（对数）+ GT 中心 + GT 预测环，直观看清到底有没有环。"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)

import pyFAI.azimuthalIntegrator
from PIL import Image, ImageDraw
import zoo_run as ZR
from torchfai.phases import phase_qs

CASES = {
    "Pilatus1M": ("Pilatus1M.edf", "Pilatus1M.poni", "AgBh"),
    "mar3450": ("LaB6_260210.mar3450", "Mar3450.poni", "LaB6"),
    "Pilatus6M": ("Pilatus6M.cbf", "Pilatus6M.poni", None),
}


def render(tag, img_f, poni_f, ph):
    img = ZR.load_img(os.path.join(ZOO, img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    # 对数压缩 + 归一化
    v = np.clip(img, 0, None)
    v = np.log1p(v)
    lo, hi = np.percentile(v, [1, 99.7])
    v = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    H, W = v.shape
    scale = max(1, int(np.ceil(max(H, W) / 900)))
    hh, ww = H // scale, W // scale
    small = v[:hh * scale, :ww * scale].reshape(hh, scale, ww, scale).mean(axis=(1, 3))
    rgb = (np.stack([small, small, small], -1) * 255).astype(np.uint8)
    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im)

    try:
        f2d = ai.getFit2D()
        gx, gy = f2d["centerX"] / scale, f2d["centerY"] / scale
    except Exception:
        gx, gy = ai.poni2 / ai.detector.pixel2 / scale, ai.poni1 / ai.detector.pixel1 / scale
    dr.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], outline=(255, 0, 0), width=2)

    if ph:
        lam = ai.wavelength
        qs = np.asarray(phase_qs(ph))
        tt = 2 * np.arcsin(np.clip((lam * 1e10) * qs / (4 * np.pi), 0, 0.9999))
        rr = ai.dist * np.tan(tt) / ai.detector.pixel1 / scale
        for r in rr:
            if 3 < r < max(hh, ww):
                dr.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(0, 200, 255), width=1)

    out = os.path.join(ROOT, "docs", f"_diag_{tag}.png")
    im.save(out)
    print(tag, "->", out, "shape", img.shape, "gt_center_px", (gx * scale, gy * scale),
          "rings", None if not ph else [round(float(x), 1) for x in rr][:10])


if __name__ == "__main__":
    for tag, (img_f, poni_f, ph) in CASES.items():
        render(tag, img_f, poni_f, ph)

