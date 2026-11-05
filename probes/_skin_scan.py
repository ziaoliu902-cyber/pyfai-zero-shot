# -*- coding: utf-8 -*-
"""跨帧皮肤特征 vs 残差：找 250/300 顽固偏差的共变特征。
特征（rebin 画布，auto_gain 后）：
  n_ring  径向剖面可检环数
  fwhm    主环半高宽 px
  az_cv   环带方位角变异系数
  sharp   |laplacian| 均值（锐度）
  pbg     环峰/背景 比
"""
import os
import sys

import numpy as np
import torch
from scipy.ndimage import laplace

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI

from real_refine_v6 import load_px_nn, match_gain
from torchfai.dataset import norm_to_param

src = open(os.path.join(ROOT, "probes", "_rebin_test.py"),
           encoding="utf-8").read()
g = {}
exec(compile(src.split("print(f\"")[0], "rebin_fn", "exec"), g)
rebin_to_canvas = g["rebin_to_canvas"]

m, dev, consts = load_px_nn(r"runs\v4w2\best.pt")
consts = g["_T"](os.path.join(ROOT, "data", "train_v4w2"), "val").norm_consts
m.eval()
LAM = 0.972386
MED_LOG_REF = 6.68
PXEFF = 834.2e-6


def auto_s(canvas):
    v = canvas[canvas > 0]
    lo, hi = 0.05, 64.0
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        if float(np.median(np.log1p(v * mid))) - MED_LOG_REF < 0:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def radial(canvas, pr, pc):
    yy, xx = np.mgrid[0:521, 0:490]
    r = np.hypot(yy - pr, xx - pc)
    idx = np.clip((r / 2).astype(int), 0, 244)
    prof = np.bincount(idx.ravel(), weights=canvas.ravel(), minlength=245)
    cnt = np.bincount(idx.ravel(), minlength=245)
    return prof / np.maximum(cnt, 1)


print(f"{'帧':>4s}{'残差mm':>8s}{'n_ring':>7s}{'fwhm':>6s}{'az_cv':>7s}"
      f"{'sharp':>8s}{'pbg':>7s}")
rows = []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    reb = np.clip(match_gain(rebin_to_canvas(img, p, 2.425)), 0, None)
    reb = reb * auto_s(reb) * 0.5
    pr = p.poni1 / PXEFF
    pc = p.poni2 / PXEFF

    x = torch.from_numpy(np.log1p(reb)[None, None].astype(np.float32)).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([LAM], device=dev),
                  torch.tensor([2.425], device=dev))
    err = float(norm_to_param(mu.double().cpu().numpy(), consts,
                              2.425)[0, 0]) * 1000 - gt

    prof = radial(reb, pr, pc)
    bg = np.median(prof)
    pk_idx = [i for i in range(8, 237)
              if prof[i] == max(prof[i - 3:i + 4]) and prof[i] > 2.5 * bg
              and (not pk_idx or i - pk_idx[-1] > 4)] if False else []
    pk_idx = []
    for i in range(8, 237):
        if prof[i] == max(prof[i - 3:i + 4]) and prof[i] > 2.5 * bg:
            if not pk_idx or i - pk_idx[-1] > 4:
                pk_idx.append(i)
    n_ring = len(pk_idx)
    if pk_idx:
        i0 = max(pk_idx, key=lambda i: prof[i])
        h = prof[i0] / 2
        l = i0
        while l > 0 and prof[l] > h: l -= 1
        r_ = i0
        while r_ < 244 and prof[r_] > h: r_ += 1
        fwhm = (r_ - l) * 2
        band_r = i0
    else:
        fwhm, band_r = 0, 100
    yy, xx = np.mgrid[0:521, 0:490]
    rr = np.hypot(yy - pr, xx - pc)
    band = (rr > band_r * 2 - 3) & (rr < band_r * 2 + 3)
    v = reb[band]
    az_cv = float(np.std(v) / (np.mean(v) + 1e-9))
    L = np.log1p(reb)
    sharp = float(np.mean(np.abs(laplace(L))))
    pbg = float(prof.max() / (bg + 1e-9))
    rows.append((d, err, n_ring, fwhm, az_cv, sharp, pbg))
    print(f"{d:>4d}{err:+8.1f}{n_ring:>7d}{fwhm:>6d}{az_cv:>7.3f}"
          f"{sharp:>8.3f}{pbg:>7.1f}")

print("\n与残差的相关性:")
E = np.array([r[1] for r in rows])
for j, nm in enumerate(["n_ring", "fwhm", "az_cv", "sharp", "pbg"], start=2):
    V = np.array([r[j] for r in rows])
    print(f"  {nm:>7s}: r={np.corrcoef(V, E)[0,1]:+.2f}")
