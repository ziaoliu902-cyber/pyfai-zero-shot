# -*- coding: utf-8 -*-
"""_tmp_rmsfloor.py：近心箱 B 臂 rms_floor 扫描——验证"宽环真解被 rms 门误杀"。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
from solver_core import solve, ring_radii
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
print(f"近心样本 {len(near)} 个，rms_floor 扫描（B 臂=几何中心+全域）：")
print(f"{'floor':>7}{'成功':>6}{'中位ΔD%':>10}")
for floor in (3.5, 6.0, 9.0, 14.0):
    rs = []
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
        pk = ring_radii(img, (W - 1) / 2, (H - 1) / 2)[0]
        s = solve(pk, (W - 1) / 2, (H - 1) / 2, 1.0 / PX, Q,
                  float(aux[5]), PX, I_rel=IR, canvas=img,
                  d_range=(0.05, 3.0), rms_floor=floor)
        if s is None or s[3] < 3:
            rs.append(np.nan)
        else:
            rs.append(abs(s[2] * PX - p[0]) / p[0] * 100)
    rs = np.array(rs)
    print(f"{floor:7.1f}{np.mean(rs < 1)*100:5.0f}%{np.nanmedian(rs):10.2f}")
