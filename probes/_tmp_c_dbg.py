# -*- coding: utf-8 -*-
"""_tmp_c_dbg.py：中段离轴 C 臂单样本 dbg——真解候选 vs 胜者逐项比分。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
import torch

from real_refine_v6 import load_px_nn
from solver_core import solve, ring_radii
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.phases import PHASES, PHASE_NAMES, phase_qs

PX = 344e-6


def bc(d, p1, p2, r1, r2):
    v2 = -d * np.tan(r1)
    v1 = np.tan(r2) * d / np.cos(r1)
    return (p2 + v2) / PX - 0.5, (p1 + v1) / PX - 0.5


ds = TorchFAIDataset(r"D:\pyfai-project\data\train_v4w3", "val")
consts = ds.norm_consts
m1, dev, _ = load_px_nn(r"runs\v4s1\best.pt")
m1.eval()
H, W = ds._mimgs.shape[1], ds._mimgs.shape[2]
sel = ds._sel
p_all = ds._mparams[sel]
aux_all = ds._maux[sel]
cx, cy = bc(p_all[:, 0], p_all[:, 1], p_all[:, 2], p_all[:, 3], p_all[:, 4])
off = np.maximum(np.abs(cx - (W - 1) / 2) / W,
                 np.abs(cy - (H - 1) / 2) / H)
mid = np.nonzero((off > 0.15) & (off < 0.45) & (aux_all[:, 1] >= 4))[0]
for j in mid[:3]:
    g = int(sel[j])
    img = np.asarray(ds._mimgs[g]).astype(np.float64)
    p = ds._mparams[g].astype(np.float64)
    aux = ds._maux[g]
    ph = PHASE_NAMES[int(aux[0])]
    Q = phase_qs(ph)
    IR = PHASES[ph][1]
    keep = IR >= 0.10
    Q, IR = Q[keep], IR[keep]
    lamA = float(aux[5])
    D_gt = p[0]
    x = torch.from_numpy(
        np.log1p(img[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m1(x, torch.tensor([lamA], device=dev),
                   torch.tensor([1.0], device=dev))
    p6 = norm_to_param(mu.double().cpu().numpy(), consts, 1.0)[0]
    cxN, cyN = bc(p6[0], p6[1], p6[2], p6[3], p6[4])
    print(f"\n== 样本 {g} 相={ph} λ={lamA:.2f} D_gt={D_gt*100:.0f}cm "
          f"NN: D={p6[0]*100:.0f}cm 环心err={np.hypot(cxN-cx[j], cyN-cy[j]):.0f}px")
    nb = max(256, int(1.05 * float(np.hypot(H, W)) / 2))
    peaks, prof, bin_px, good = ring_radii(img, cxN, cyN, nb=nb)
    s = solve(peaks, cxN, cyN, p6[0] / PX, Q, lamA, PX, I_rel=IR,
              canvas=img, prof_ctx=(prof, bin_px, good),
              D_ref_m=float(p6[0]), d_range=(0.35, 2.8), dbg=True)
    if s:
        print(f"胜者: D={s[2]*PX*100:.1f}cm (真 {D_gt*100:.0f}) "
              f"m={s[3]} rms={s[4]:.2f} pscore={s[6]:+.3f}")
