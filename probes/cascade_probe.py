# -*- coding: utf-8 -*-
"""cascade_probe.py：F 架构（geo+RS+ray×3+NN 多种子级联）在生产画布上的探针。

目的（不改生产代码）：对 ceria 七档逐帧跑所有种子，打印每种子解的质量与
与 GT 的误差，比较不同质量门的择优选谁，为 solve_auto 生产化定门。

用法: python probes/cascade_probe.py
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI
import torch

from real_refine_v6 import load_px_nn, to_nn_input_wide
from solver_core import (center_radial_symmetry, ring_radii, solve,
                         to_canvas)
from torchfai.dataset import TorchFAIDataset as _T
from torchfai.dataset import norm_to_param
from torchfai.phases import phase_qs

LAM = 0.972386
PXEFF = 172e-6 / (521.0 / 2527)
Q = phase_qs("CeO2")[:8]
ds = _T(os.path.join(ROOT, "data", "train_v4w3"), "val")
consts = ds.norm_consts
m1, dev, _ = load_px_nn(r"runs\v4s1\best.pt")
m1.eval()


def solve_at(canvas, cx, cy, D_px0, px_m, d_range, D_ref=0.0,
             rms_floor=1.8):
    peaks, prof, bin_px, good = ring_radii(canvas, cx, cy)
    return solve(peaks, cx, cy, D_px0, Q, LAM, px_m,
                 d_range=d_range, rms_floor=rms_floor,
                 canvas=canvas, prof_ctx=(prof, bin_px, good),
                 D_ref_m=D_ref)


def show(tag, sol, gt, pxm):
    if sol is None:
        print(f"    {tag:>6}  失败")
        return None
    cx, cy, D, m, rms, icorr, ps = sol
    dmm = D * pxm * 1000
    print(f"    {tag:>6} D={dmm:7.1f} Δ={dmm - gt:+6.1f} m={m} "
          f"rms={rms:4.2f} Icorr={icorr:+.2f} ps={ps:+.3f} "
          f"中心=({cx:5.1f},{cy:5.1f})")
    return dmm


print("== F 级联探针：ceria 七档（wide 画布）==")
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    wide, k, extra = to_nn_input_wide(img)
    Hm, Wm = wide.shape
    pxm = PXEFF * 521.0 / Hm
    print(f"\n--- ceria {d}mm (GT {gt:.1f}mm) wide={wide.shape} ---")

    # B' 参考：to_canvas 朴素画布经典
    canvas, s, cx0, cy0 = to_canvas(img)
    pxB = 172e-6 * s
    solBp = solve_at(canvas, cx0, cy0, 1.0 / pxB, pxB, (0.05, 3.0))
    if solBp:
        print(f"    [B']  D={solBp[2] * pxB * 1000:7.1f} "
              f"Δ={solBp[2] * pxB * 1000 - gt:+6.1f} m={solBp[3]} "
              f"rms={solBp[4]:.2f}")

    # NN 先验
    x = torch.from_numpy(np.log1p(wide[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m1(x, torch.tensor([LAM], device=dev),
                   torch.tensor([k], device=dev))
    p6 = norm_to_param(mu.double().cpu().numpy(), consts, k)[0]
    cxN = p6[2] / PXEFF * (Wm / 490.0)
    cyN = p6[1] / PXEFF * (Hm / 521.0)
    Dp0 = p6[0] / PXEFF * (Hm / 521.0)

    geo = ((Wm - 1) / 2.0, (Hm - 1) / 2.0)
    rs = center_radial_symmetry(wide)
    print(f"    geo=({geo[0]:.1f},{geo[1]:.1f}) rs=({rs[0]:.1f},{rs[1]:.1f}) "
          f"nn=({cxN:.1f},{cyN:.1f})")
    cands = []
    seeds = [("geo", *geo), ("rs", *rs)]
    for t in (1.0, 2.0, 3.5):
        seeds.append((f"ray{t}", rs[0] + t * (rs[0] - geo[0]),
                      rs[1] + t * (rs[1] - geo[1])))
    for tag, sx, sy in seeds:
        sol = solve_at(wide, sx, sy, 1.0 / pxm, pxm, (0.05, 3.0))
        dmm = show(tag, sol, gt, pxm)
        if sol and sol[3] >= 3:
            cands.append((tag, sol, dmm))

    solN = solve_at(wide, cxN, cyN, Dp0, pxm, (0.35, 2.8), D_ref=p6[0])
    dmmN = show("nn", solN, gt, pxm)
    if solN and solN[3] >= 3:
        cands.append(("nn", solN, dmmN))

    if cands:
        bm = sorted(cands, key=lambda t: (-t[1][3], -t[1][6], t[1][4]))[0]
        print(f"    >> 门(-m,-ps,rms): {bm[0]} Δ={bm[2] - gt:+.1f}mm"
              f"   [候选 {', '.join(t[0] for t in cands)}]")
