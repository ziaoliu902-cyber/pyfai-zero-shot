# -*- coding: utf-8 -*-
"""_tmp_dbg_near.py：单样本全链路 dbg——近心 B 臂失败解剖。"""
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
near = np.nonzero((off < 0.05) & (aux_all[:, 1] >= 4))[0]
j = int(near[0])
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
print(f"样本: 相={ph} λ={lamA:.3f}Å D_gt={p[0]*100:.0f}cm "
      f"环心=({cx[j]:.0f},{cy[j]:.0f}) off={off[j]:.3f} n_vis={aux[1]:.0f}")
Dpx_gt = p[0] / PX
rm_gt = _r_model(Q, lamA, Dpx_gt)
print(f"GT 模型环半径(px): {np.round(rm_gt,0)}")
pk = ring_radii(img, (W - 1) / 2, (H - 1) / 2)[0]
print(f"ring_radii@几何中心 峰数={len(pk)}: "
      f"{[(round(r), round(h)) for r, h in pk]}")
s = solve(pk, (W - 1) / 2, (H - 1) / 2, 1.0 / PX, Q, lamA, PX,
          I_rel=IR, canvas=img, d_range=(0.05, 3.0), rms_floor=3.5,
          dbg=True)
print("解:", "None" if s is None else
      f"D={s[2]*PX*100:.1f}cm m={s[3]} rms={s[4]:.2f} cx,cy=({s[0]:.0f},{s[1]:.0f})")
