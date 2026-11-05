# -*- coding: utf-8 -*-
"""validate_poni.py：全几何终验——拟合 poni vs GT poni 的全探测器 2θ 对拍。

用法: python probes/validate_poni.py
对 ceria 七档 + alumina_mean 逐对计算 2θ 差（mrad）的 mean/rms/max。
"""
import glob
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import fabio
import pyFAI


def load_poni(path):
    """载入 poni；SplineFile 为相对路径时补全。"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    lines = []
    base = os.path.dirname(path)
    for ln in txt.splitlines():
        if ln.startswith("SplineFile:"):
            sp = ln.split(":", 1)[1].strip()
            if sp.lower() in ("none", ""):
                continue
            if not os.path.isabs(sp):
                ln = "SplineFile: " + os.path.join(base, sp)
        lines.append(ln)
    tmp = os.path.join(os.environ["TEMP"], "_val_tmp.poni")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return pyFAI.load(tmp)


def grid_tth(ai, shape, stride=8):
    H, W = shape
    d1 = np.arange(0, H, stride, dtype=np.float64)
    d2 = np.arange(0, W, stride, dtype=np.float64)
    D1, D2 = np.meshgrid(d1, d2, indexing="ij")
    return ai.tth(D1, D2)


def stats(name, t_fit, t_gt):
    d = (t_fit - t_gt) * 1e3   # mrad
    print(f"{name:>22}  mean={d.mean():+6.2f}  rms={np.sqrt((d**2).mean()):6.2f}"
          f"  max|.|={np.abs(d).max():6.2f} mrad   (n={d.size})")


print("== 全几何 2θ 对拍（拟合 poni vs GT）==")
for d in (150, 200, 250, 300, 350, 400, 450):
    fit = os.path.join(ROOT, "probes", "results", f"ceria_{d}.poni")
    gt = os.path.join(ROOT, "data", "real", "MX_ceria",
                      f"ceria_{d}_1_0001.poni")
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    af, ag = load_poni(fit), load_poni(gt)
    stats(f"ceria_{d}", grid_tth(af, img.shape), grid_tth(ag, img.shape))

fit = os.path.join(ROOT, "probes", "results", "alumina_mean.poni")
gt = glob.glob(os.path.join(ROOT, "data", "real", "alumina", "*.poni"))[0]
img = fabio.open(sorted(glob.glob(
    os.path.join(ROOT, "data", "real", "alumina", "*.edf")))[0]).data
af, ag = load_poni(fit), load_poni(gt)
stats("alumina_mean", grid_tth(af, img.shape), grid_tth(ag, img.shape))
