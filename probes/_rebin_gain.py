# -*- coding: utf-8 -*-
"""rebin 画布 × 增益扫描：几何修好后，亮度是剩下的杠杆吗？"""
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
code = src.split("print(f\"")[0]
g = {}
exec(compile(code, "rebin_fn", "exec"), g)
rebin_to_canvas = g["rebin_to_canvas"]

m, dev, consts = load_px_nn(r"runs\v4w2\best.pt")
consts = g["_T"](os.path.join(ROOT, "data", "train_v4w2"), "val").norm_consts
m.eval()
LAM = 0.972386
SW = [0.5, 1, 2, 4, 8, 16]

print("帧\\s " + "".join(f"{s:>8.1f}" for s in SW))
best = {}
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    reb = np.clip(match_gain(rebin_to_canvas(img, p, 2.425)), 0, None)
    row = f"{d:>5d} "
    bb = (1e9, None)
    for s in SW:
        x = torch.from_numpy(
            np.log1p(reb * s)[None, None].astype(np.float32)).to(dev)
        with torch.no_grad():
            mu, _ = m(x, torch.tensor([LAM], device=dev),
                      torch.tensor([2.425], device=dev))
        e = float(norm_to_param(mu.double().cpu().numpy(), consts,
                                2.425)[0, 0]) * 1000 - gt
        row += f"{e:+8.1f}"
        if abs(e) < abs(bb[0]):
            bb = (e, s)
    best[d] = bb
    print(row)
print("\n帧级 s*:", {d: (v[1], round(v[0], 1)) for d, v in best.items()})
