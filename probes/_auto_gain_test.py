# -*- coding: utf-8 -*-
"""auto_gain 落地：median(log1p(非零)) 对齐训练域 → 自动 s。
7 帧终审 + 自动值附近 ×{0.5,1,1.5,2} 平坦度检查。
"""
import os
import sys

import numpy as np
import torch

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

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default=r"runs\v4w2\best.pt")
_a = _ap.parse_args()
m, dev, consts = load_px_nn(_a.model)
consts = g["_T"](os.path.join(ROOT, "data", "train_v4w3"), "val").norm_consts
m.eval()
LAM = 0.972386
MED_LOG_REF = 6.68        # 训练域 median(log1p(非零))（_gain_calib 实测）


def auto_s(canvas):
    v = canvas[canvas > 0]
    target = MED_LOG_REF

    def err(s):
        return float(np.median(np.log1p(v * s))) - target
    lo, hi = 0.05, 64.0
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        if err(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


MULT = [0.5, 1.0, 1.5, 2.0]
print(f"{'帧':>5s}{'auto_s':>8s}" + "".join(f"{f'×{x}':>9s}" for x in MULT))
errs = {x: [] for x in MULT}
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    reb = np.clip(match_gain(rebin_to_canvas(img, p, 2.425)), 0, None)
    s0 = auto_s(reb)
    row = f"{d:>5d}{s0:>8.2f}"
    for x in MULT:
        xx = torch.from_numpy(np.log1p(reb * s0 * x)[None, None].astype(
            np.float32)).to(dev)
        with torch.no_grad():
            mu, _ = m(xx, torch.tensor([LAM], device=dev),
                      torch.tensor([2.425], device=dev))
        e = float(norm_to_param(mu.double().cpu().numpy(), consts,
                                2.425)[0, 0]) * 1000 - gt
        errs[x].append(abs(e))
        row += f"{e:+9.1f}"
    print(row)
print("\n平均|Δ|:", {f"×{x}": round(float(np.mean(v)), 1)
                    for x, v in errs.items()})
