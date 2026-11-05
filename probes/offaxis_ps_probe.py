# -*- coding: utf-8 -*-
"""offaxis_ps_probe.py：离轴合成样本上多种子解的 pscore 行为抽查。

只回答一个问题：在 0.15~2.5 离轴域，真解（RS/射线/NN）的 pscore 是否
仍高于伪解？若 pscore-window 门在离轴域也成立，则可安全用于生产。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import torch

from real_refine_v6 import load_px_nn
from solver_core import center_radial_symmetry, ring_radii, solve
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.phases import PHASES, PHASE_NAMES, phase_qs

PX = 344e-6
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BINS = [(0.15, 0.45), (0.45, 2.5)]


def bc(d, p1, p2, r1, r2):
    v2 = -d * np.tan(r1)
    v1 = np.tan(r2) * d / np.cos(r1)
    return (p2 + v2) / PX - 0.5, (p1 + v1) / PX - 0.5


ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v4w3"), "val")
consts = ds.norm_consts
Hm, Wm = ds._mimgs.shape[1], ds._mimgs.shape[2]
m1, dev, _ = load_px_nn(r"runs\v4s1\best.pt")
m1.eval()

picks = {b: [] for b in BINS}
rng = np.random.default_rng(11)
for i in rng.permutation(len(ds)):
    g = int(ds._sel[i])
    p = ds._mparams[g]
    aux = ds._maux[g]
    if aux[1] < 4 or abs(p[3]) > 0.02 or abs(p[4]) > 0.02 \
            or not (0.8 <= aux[5] <= 1.6):
        continue
    ph = PHASE_NAMES[int(aux[0])]
    Q0 = phase_qs(ph)
    IR0 = PHASES[ph][1]
    k0 = IR0 >= 0.10
    rm0 = (Q0[k0] * 0 + 1) * 0  # placeholder
    from solver_core import _r_model
    rm0 = _r_model(Q0[k0], float(aux[5]), float(p[0]) / PX)
    if ((rm0 > 25) & (rm0 < min(Hm, Wm) * 0.707)).sum() < 5:
        continue
    cx_gt, cy_gt = bc(p[0], p[1], p[2], p[3], p[4])
    off = max(abs(cx_gt - (Wm - 1) / 2) / Wm,
              abs(cy_gt - (Hm - 1) / 2) / Hm)
    for b in BINS:
        if b[0] <= off < b[1] and len(picks[b]) < N:
            picks[b].append(i)
            break
    if all(len(v) >= N for v in picks.values()):
        break


def solve_at(img, cx, cy, D_px0, Q, IR, lamA, D_ref=0.0, d_range=(0.05, 3.0)):
    nb = max(256, int(1.05 * float(np.hypot(*img.shape)) / 2))
    peaks, prof, bin_px, good = ring_radii(img, cx, cy, nb=nb)
    return solve(peaks, cx, cy, D_px0, Q, lamA, PX, I_rel=IR,
                 canvas=img, prof_ctx=(prof, bin_px, good),
                 D_ref_m=D_ref, d_range=d_range)


for b in BINS:
    print(f"\n== 离轴箱 {b[0]}~{b[1]} ==")
    for i in picks[b]:
        g = int(ds._sel[i])
        img = np.asarray(ds._mimgs[g]).astype(np.float64)
        p = ds._mparams[g].astype(np.float64)
        aux = ds._maux[g]
        ph = PHASE_NAMES[int(aux[0])]
        Q = phase_qs(ph)
        IR = PHASES[ph][1]
        keep = IR >= 0.10
        Q, IR = Q[keep], IR[keep]
        lamA = float(aux[5])
        D_gt = float(p[0])
        x = torch.from_numpy(np.log1p(img[None, None].astype(np.float32))).to(dev)
        with torch.no_grad():
            mu, _ = m1(x, torch.tensor([lamA], device=dev),
                       torch.tensor([1.0], device=dev))
        p6 = norm_to_param(mu.double().cpu().numpy(), consts, 1.0)[0]
        cxN, cyN = bc(p6[0], p6[1], p6[2], p6[3], p6[4])
        geo = ((Wm - 1) / 2, (Hm - 1) / 2)
        rs = center_radial_symmetry(img)
        seeds = [("geo", *geo), ("rs", *rs)]
        for t in (1.0, 2.0, 3.5):
            seeds.append((f"ray{t}", rs[0] + t * (rs[0] - geo[0]),
                          rs[1] + t * (rs[1] - geo[1])))
        print(f"\n sample {i} {ph} D_gt={D_gt*1000:.1f}mm off="
              f"{max(abs(bc(p[0],p[1],p[2],p[3],p[4])[0]-(Wm-1)/2)/Wm, abs(bc(p[0],p[1],p[2],p[3],p[4])[1]-(Hm-1)/2)/Hm):.2f}")
        for tag, sx, sy in seeds:
            s = solve_at(img, sx, sy, 1.0 / PX, Q, IR, lamA)
            if s is None:
                print(f"   {tag:>6} 失败")
                continue
            err = abs(s[2] * PX - D_gt) / D_gt * 100
            print(f"   {tag:>6} ΔD={err:5.1f}% m={s[3]} ps={s[6]:+.3f} "
                  f"rms={s[4]:.2f} 中心=({s[0]:.0f},{s[1]:.0f})")
        sN = solve_at(img, cxN, cyN, p6[0] / PX, Q, IR, lamA,
                      D_ref=float(p6[0]), d_range=(0.35, 2.8))
        if sN:
            err = abs(sN[2] * PX - D_gt) / D_gt * 100
            print(f"   {'nn':>6} ΔD={err:5.1f}% m={sN[3]} ps={sN[6]:+.3f} "
                  f"rms={sN[4]:.2f} 中心=({sN[0]:.0f},{sN[1]:.0f})")

