# -*- coding: utf-8 -*-
"""_tmp_consensus_clean.py：共识门·干净域判决（ceria 七档全种子）。
判据：共识门选中的解 |ΔD| ≤ 3mm（=classic-first 回归线），且不选
射线伪解（doc11 §2.1 的 m-first 反例必须被翻案）。"""
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
from solver_core import (center_radial_symmetry, consensus_pick,
                         ring_radii, solve, to_canvas)
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

# 注意：doc11 生产路径用全帧 to_canvas（非 wide），此处与 cascade_probe
# 相同用 wide 画布——判决对象是"门"而非画布，两域同构即可。


def solve_at(canvas, cx, cy, D_px0, px_m, d_range, D_ref=0.0):
    peaks, prof, bin_px, good = ring_radii(canvas, cx, cy)
    return solve(peaks, cx, cy, D_px0, Q, LAM, px_m,
                 d_range=d_range, rms_floor=1.8,
                 canvas=canvas, prof_ctx=(prof, bin_px, good),
                 D_ref_m=D_ref)


print("== 共识门·干净域判决：ceria 七档（wide 画布，全种子）==")
print(f"{'帧':>6}{'候选(种子:Δmm:票)':>52}{'共识选':>8}{'Δmm':>8}"
      f"{'m-first 选':>12}{'Δmm':>9}")
bad = 0
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    wide, k, extra = to_nn_input_wide(img)
    Hm, Wm = wide.shape
    pxm = PXEFF * 521.0 / Hm
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
    seeds = [("geo", *geo), ("rs", *rs)]
    for t in (1.0, 2.0, 3.5):
        seeds.append((f"ray{t}", rs[0] + t * (rs[0] - geo[0]),
                      rs[1] + t * (rs[1] - geo[1])))
    cands = []
    for tag, sx, sy in seeds:
        sol = solve_at(wide, sx, sy, 1.0 / pxm, pxm, (0.05, 3.0))
        if sol and sol[3] >= 3:
            cands.append((tag, sol))
    solN = solve_at(wide, cxN, cyN, Dp0, pxm, (0.35, 2.8), D_ref=p6[0])
    if solN and solN[3] >= 3:
        cands.append(("nn", solN))
    if not cands:
        print(f"{d:>6}  全部失败")
        bad += 1
        continue
    # 共识门
    cp = consensus_pick(cands)
    tagC, solC, votesC = cp
    dC = solC[2] * pxm * 1000 - gt
    # 旧 m-first 门（对照）
    tagM, solM = sorted(cands, key=lambda t: (-t[1][3], -t[1][6],
                                              t[1][4]))[0]
    dM = solM[2] * pxm * 1000 - gt
    votes_map = {}
    # 重算每候选票数用于展示
    for i, (ti, si) in enumerate(cands):
        v = 1
        for j, (tj, sj) in enumerate(cands):
            if j != i and (abs(si[2] - sj[2]) / max(si[2], 1e-9) < 0.02
                           and np.hypot(si[0] - sj[0], si[1] - sj[1]) < 12):
                v += 1
        votes_map[ti] = v
    disp = " ".join(f"{t}:{si[2]*pxm*1000-gt:+.0f}:{votes_map[t]}"
                    for t, si in cands)
    ok = abs(dC) <= 3.0
    if not ok:
        bad += 1
    print(f"{d:>6}{disp:>52}{tagC:>8}{dC:+8.1f}{tagM:>12}{dM:+9.1f}"
          f"{'✓' if ok else '✗ ✗ ✗'}")
print(f"\n判决：{'共识门通过（≤3mm 全档）' if bad == 0 else f'{bad} 档失败'}")
