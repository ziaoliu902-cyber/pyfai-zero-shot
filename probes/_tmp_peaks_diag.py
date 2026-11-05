# -*- coding: utf-8 -*-
"""_tmp_peaks_diag.py：nb=384 下逐样本诊断——剩余 ~16% 误差解剖。
看：①真环命中几个 ②假峰剩几个 ③D_fit/D_gt 比值 ④倾角大小。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
from solver_core import solve, ring_radii, _r_model
from torchfai.dataset import TorchFAIDataset
from torchfai.phases import PHASES, PHASE_NAMES, phase_qs

PX = 344e-6


def bc(d, p1, p2, r1, r2):
    v2 = -d * np.tan(r1)
    v1 = np.tan(r2) * d / np.cos(r1)
    return (p2 + v2) / PX - 0.5, (p1 + v1) / PX - 0.5


ds = TorchFAIDataset(r"D:\pyfai-project\data\train_v4w3", "val")
H, W = ds._mimgs.shape[1], ds._mimgs.shape[2]
sel = ds._sel
p_all = ds._mparams[sel]
aux_all = ds._maux[sel]
cx, cy = bc(p_all[:, 0], p_all[:, 1], p_all[:, 2], p_all[:, 3], p_all[:, 4])
off = np.maximum(np.abs(cx - (W - 1) / 2) / W,
                 np.abs(cy - (H - 1) / 2) / H)
near = np.nonzero((off < 0.05) & (aux_all[:, 1] >= 4))[0][:12]
print(f"{'相':>7}{'λ':>6}{'rot12':>14}{'GT环(画布内)':>16}"
      f"{'峰数':>5}{'真命中':>6}{'Dfit/Dgt':>10}{'m':>3}")
for j in near:
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
    rm_gt = _r_model(Q, lamA, p[0] / PX)
    vis_gt = rm_gt[(rm_gt > 25) & (rm_gt < min(H, W) * 0.707)]
    pk = ring_radii(img, (W - 1) / 2, (H - 1) / 2, nb=384)[0]
    rps = np.array([r for r, _ in pk])
    hit = sum(1 for r in vis_gt if (np.abs(rps - r) < 6).any())
    s = solve(pk, (W - 1) / 2, (H - 1) / 2, 1.0 / PX, Q, lamA, PX,
              I_rel=IR, canvas=img, d_range=(0.05, 3.0), rms_floor=3.5)
    ratio = (s[2] * PX / p[0]) if s else np.nan
    m = s[3] if s else -1
    print(f"{ph:>7}{lamA:6.2f}"
          f"({p[3]:+5.2f},{p[4]:+5.2f})"
          f"{len(vis_gt):>9}环/{f'{np.round(vis_gt,0)}':>6}"
          f"{len(pk):>5}{hit:>6}{ratio:10.3f}{m:>3}")
