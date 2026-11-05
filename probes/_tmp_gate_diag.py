# -*- coding: utf-8 -*-
"""_tmp_gate_diag.py：真环为什么没进峰表——逐环过三道门(prom/bg/dist)统计。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
from scipy.signal import find_peaks

from solver_core import _r_model, ring_radii
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
print(f"门诊断（nb={nb}, 选样: 近心+低倾角+画布内≥4真环）")
print(f"{'样本':>4}{'相':>8}{'环':>3}{'峰':>3}{'命中':>4}"
      f"{'prom漏':>6}{'bg漏':>5}{'dist漏':>6}{'裂峰':>4}")
n_ok_samples = 0
stats = []
checked = 0
for j in np.nonzero((off < 0.05) & lowtilt & (aux_all[:, 1] >= 4))[0]:
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
    if len(vis) < 4 or checked >= 12:
        continue
    checked += 1
    img = np.asarray(ds._mimgs[g]).astype(np.float64)
    pk, prof, bin_px, good = ring_radii(img, (W - 1) / 2, (H - 1) / 2,
                                        nb=nb)
    rps = np.array([r for r, _ in pk]) if pk else np.array([])
    bg = float(np.median(prof[prof > -1e18]))
    proms, props = find_peaks(prof, prominence=0, distance=1)
    hit = 0
    prom_miss = bg_miss = dist_miss = split = 0
    for r in vis:
        near = np.abs(rps - r) < 6
        if near.any():
            hit += int(near.sum() == 1)
            split += int(near.sum() > 1)
            continue
        b = int(round(r / bin_px - 0.5))
        # 在 find_peaks 全集(prominence=0)里找最近峰
        cand = [q for q in proms if abs((q + 0.5) * bin_px - r) < 8]
        hgt_ok = prof[b] > bg if 0 <= b < nb else False
        if not cand:
            if hgt_ok:
                prom_miss += 1        # 谱上有东西但连 prominence=0 峰都不是
            else:
                bg_miss += 1          # 该处谱值不高于背景 → 谱里根本没环
        else:
            dist_miss += 1            # 有峰但被 distance/子峰合并影响？
    stats.append((hit, len(vis), len(pk), prom_miss, bg_miss, dist_miss,
                  split, ph))
    n_ok_samples += 1
for i, (hit, nv, npk, pm, bm, dm, sp, ph) in enumerate(stats):
    print(f"{i:>4}{ph:>8}{nv:>3}{npk:>3}{hit:>4}{pm:>6}{bm:>5}{dm:>6}{sp:>4}")
tot = np.array([s[:1] + (0,) for s in stats]).sum()
agg = np.array([(s[0], s[1]) for s in stats]).sum(axis=0)
print(f"\n总体: 命中 {agg[0]}/{agg[1]} 环 = {agg[0]/max(agg[1],1)*100:.0f}%")
agg2 = np.array([(s[3], s[4], s[5], s[6]) for s in stats]).sum(axis=0)
print(f"未命中原因: prom漏={agg2[0]} bg漏(谱无环)={agg2[1]} "
      f"dist漏={agg2[2]} 裂峰={agg2[3]}")
