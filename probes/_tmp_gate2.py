# -*- coding: utf-8 -*-
"""_tmp_gate2.py：PROM vs DISTANCE 分项判责。
对每个真环：跑 find_peaks 两次（prom=PROM·max,dist=1）和（prom,dist=12），
看真环峰各在哪一步消失。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
from scipy.signal import find_peaks

from solver_core import MIN_R, NB, PROM, _r_model
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
lowtilt = (np.abs(p_all[:, 3]) < 0.03) & (np.abs(p_all[:, 4]) < 0.03)
nb = 384
print(f"PROM/DIST 分项判责（nb={nb}）：")
c_prom = c_dist = c_ok = c_gone = 0
for j in np.nonzero((off < 0.05) & lowtilt & (aux_all[:, 1] >= 4))[0][:12]:
    g = int(sel[j])
    p = ds._mparams[g].astype(np.float64)
    aux = ds._maux[g]
    ph = PHASE_NAMES[int(aux[0])]
    Q = phase_qs(ph)
    IR = PHASES[ph][1]
    keep = IR >= 0.10
    Q, IR = Q[keep], IR[keep]
    rm_gt = _r_model(Q, float(aux[5]), p[0] / PX)
    vis = rm_gt[(rm_gt > 25) & (rm_gt < min(H, W) * 0.707)]
    if len(vis) < 4:
        continue
    img = np.asarray(ds._mimgs[g]).astype(np.float64)
    # 复刻 ring_radii 的谱
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    r = np.hypot(yy - (H - 1) / 2, xx - (W - 1) / 2)
    rmax = float(r.max())
    idx = (r / rmax * (nb - 1)).astype(np.int32).ravel()
    v = img.ravel()
    cnt = np.bincount(idx, minlength=nb)
    prof = np.bincount(idx, weights=v, minlength=nb) / np.maximum(cnt, 1)
    bg = float(np.median(prof[cnt > 0]))
    P1, _ = find_peaks(prof, prominence=PROM * prof.max(), distance=1)
    P2, _ = find_peaks(prof, prominence=PROM * prof.max(), distance=12)
    r1 = (P1 + 0.5) * (rmax / nb)
    r2 = (P2 + 0.5) * (rmax / nb)
    for rr in vis:
        in1 = np.abs(r1 - rr) < 6
        in2 = np.abs(r2 - rr) < 6
        if in2.any():
            c_ok += 1
        elif in1.any():
            c_dist += 1            # prominence 过了，被 distance 挤掉
        else:
            b = int(round(rr / (rmax / nb) - 0.5))
            hv = prof[max(b - 2, 0):b + 3].max()
            if hv > bg:
                c_prom += 1        # 谱高于背景但 prominence 不足
            else:
                c_gone += 1        # 谱里压根没环（生成端 DW/amp 吃掉）
n_tot = c_ok + c_dist + c_prom + c_gone
print(f"真环 {n_tot} 个: 两门全过={c_ok}  distance挤掉={c_dist}  "
      f"prominence不足={c_prom}  谱无环(DW/amp)={c_gone}")
