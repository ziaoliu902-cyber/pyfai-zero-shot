# -*- coding: utf-8 -*-
"""ring_fit.py：B 路线——经典峰位几何解（ceria 真实帧验收）。

流程：NN 粗解(v4s1) 给 (cx,cy,D) 初值 → 径向谱找环半径（抛物线内插）
→ 与 CeO2 hkl q 值匹配 → 最小二乘精化 (cx,cy,D) → Δdist vs GT。
λ 已知 0.972386Å；忽略倾角（rot 小、椭圆度影响 <1% 量级，先验证主链路）。
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
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from real_refine_v6 import load_px_nn, to_nn_input_wide
from torchfai.dataset import TorchFAIDataset as _T
from torchfai.dataset import norm_to_param

LAM = 0.972386
K = 2.425
PXEFF = 172e-6 / (521.0 / 2527)          # 粗解视场等效像素(m)
CEO2_A = 5.4113                           # 晶格常数(Å)
HKLS = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1),
        (2, 2, 2), (4, 0, 0), (3, 3, 1), (4, 2, 0)]
Q_HKL = np.array([2 * np.pi / (CEO2_A / np.sqrt(h * h + k * k + l * l))
                  for h, k, l in HKLS])
NB = 2048
MIN_R = 25.0

ds = _T(os.path.join(ROOT, "data", "train_v4w3"), "val")
consts = ds.norm_consts
m1, dev, _ = load_px_nn(r"runs\v4s1\best.pt"); m1.eval()


def nn_coarse(img):
    wide, k, _ = to_nn_input_wide(img)
    x = torch.from_numpy(np.log1p(wide[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m1(x, torch.tensor([LAM], device=dev),
                   torch.tensor([k], device=dev))
    p6 = norm_to_param(mu.double().cpu().numpy(), consts, k)[0]
    return p6, wide, k          # [dist_m, poni1, poni2, r1, r2, r3]


def ring_radii(canvas, cx, cy):
    """均值径向谱（中位数实验教训：环像素也是少数，median 把环一起杀了）
    + 抛物线内插峰位。"""
    H, W = canvas.shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    r = np.hypot(yy - cy, xx - cx)
    rmax = float(r.max())
    idx = (r / rmax * (NB - 1)).astype(np.int32).ravel()
    v = canvas.astype(np.float64).ravel()
    cnt = np.bincount(idx, minlength=NB)
    prof = np.bincount(idx, weights=v, minlength=NB) / np.maximum(cnt, 1)
    pk, _ = find_peaks(prof, prominence=0.05 * prof.max(), distance=12)
    out = []
    bin_px = rmax / NB
    for p in pk:
        if p <= 2 or p >= NB - 3:
            continue
        y0, y1, y2 = prof[p - 1], prof[p], prof[p + 1]
        d = 0.5 * (y0 - y2) / max(y0 - 2 * y1 + y2, 1e-12)   # 抛物线顶点
        rp = (p + 0.5 + np.clip(d, -1, 1)) * bin_px
        if rp < MIN_R:
            continue
        out.append((rp, prof[p]))
    out.sort()
    return out


def r_model(q, cx, cy, D_px):       # 环半径(px, 粗解 canvas 坐标)
    tt = 2 * np.arcsin(np.clip(q * LAM / (4 * np.pi), 0, 0.999))
    return D_px * np.tan(tt)


def solve(peaks_rp, cx0, cy0, D_px0, dbg=False, D_nn_m=0.0, px_m=PXEFF):
    """鲁棒三段解 + NN 盆域门（400档教训：伪影簇可意外自洽成解，9票假解
    击退10票真解——必须用 NN 粗解定盆：解须落在 NN ±80mm 内）：
    ① D 投票（前2大簇多假设）② 比例容差迭代配对（3%→2%→1%→0.5%）
    ③ LSQ 精化 + 盆域门。"""
    votes = []
    for rp, _ in peaks_rp:
        for q in Q_HKL:
            tt = 2 * np.arcsin(np.clip(q * LAM / (4 * np.pi), 0, 0.999))
            D = rp / np.tan(tt)
            if 0.35 * D_px0 < D < 2.8 * D_px0:
                votes.append(D)
    if len(votes) < 3:
        if dbg:
            print("    [dbg] 投票<3")
        return None
    votes = np.sort(np.array(votes))
    clusters = []
    used = np.zeros(len(votes), bool)
    for i in range(len(votes)):
        if used[i]:
            continue
        j = i
        while j < len(votes) and votes[j] < votes[i] * 1.03:
            j += 1
        clusters.append((j - i, float(np.mean(votes[i:j]))))
        used[i:j] = True
    clusters.sort(reverse=True)
    clusters = [c for c in clusters if c[0] >= 3][:3]
    if not clusters:
        if dbg:
            print("    [dbg] 无≥3票簇")
        return None

    def pair_up(cx, cy, D, frac):
        pairs = []
        for (h_, k_, l_), q in zip(HKLS, Q_HKL):
            rm = r_model(np.array([q]), cx, cy, D)[0]
            tol = max(1.5, frac * rm)
            c = [(abs(rp - rm), rp) for rp, _ in peaks_rp if abs(rp - rm) < tol]
            if c:
                pairs.append((q, min(c)[1]))
        return pairs

    def refine(D_star):
        """逐轮（3%→2%→1%）配对+LSQ，快照最优（400/450教训：最紧轮次
        可能把分裂峰挤出门，不能以最后一轮成败定生死）。"""
        cx, cy, D = cx0, cy0, D_star
        best = None
        for frac in (0.03, 0.02, 0.01):
            pairs = pair_up(cx, cy, D, frac)
            if len(pairs) >= 3:
                qs = np.array([q for q, _ in pairs])
                rps = np.array([rp for _, rp in pairs])

                def resid(p):
                    return r_model(qs, p[0], p[1], p[2]) - rps

                s = least_squares(resid, [cx, cy, D], method="lm")
                cx, cy, D = s.x
                rms = float(np.sqrt(np.mean(s.fun ** 2)))
                if best is None or (len(pairs), -rms) > (best[3], -best[4]):
                    best = (float(s.x[0]), float(s.x[1]), float(s.x[2]),
                            len(pairs), rms, D_star)
        return best

    cands = []
    for n_votes, D_star in clusters:
        r = refine(D_star)
        if r is None:
            if dbg:
                print(f"    [dbg] 簇 D*={D_star:.1f}({n_votes}票) 配对<3")
            continue
        cx, cy, D, m, rms, Ds = r
        basin_ok = (D_nn_m <= 0 or abs(D * px_m - D_nn_m) < 0.08)
        if dbg:
            print(f"    [dbg] 簇 D*={Ds:.1f}({n_votes}票) 配 {m} 环 "
                  f"rms={rms:.2f}px D_fit={D:.1f} 盆域={'✓' if basin_ok else '✗'}")
        if not basin_ok:
            continue
        if m == 3 and abs(D - Ds) / Ds > 0.03:
            continue
        if m >= 4 and rms > 1.8:
            continue
        cands.append((m, n_votes, -rms, cx, cy, D))
    if not cands:
        return None
    cands.sort(reverse=True)          # 环数↓ 票数↓ rms↑
    m, nv, neg_rms, cx, cy, D = cands[0]
    return (cx, cy, D, m, -neg_rms)


print(f"{'帧':>5}{'NNΔmm':>9}{'fitΔmm':>8}{'环数':>5}{'rms_px':>8}{'D_px':>9}")
nn_es, fit_es = [], []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    p6, wide, k = nn_coarse(img)
    nn_e = p6[0] * 1000 - gt
    # 粗解坐标: poni 在 canvas 系 → cx,cy(px)；D 换算到 canvas 像素
    Hm, Wm = wide.shape
    cx0 = p6[2] / PXEFF * (Wm / 490.0)
    cy0 = p6[1] / PXEFF * (Hm / 521.0)
    D_px0 = p6[0] / PXEFF * (Hm / 521.0)
    peaks = ring_radii(wide, cx0, cy0)
    sol = solve(peaks, cx0, cy0, D_px0, dbg=True, D_nn_m=p6[0],
                px_m=PXEFF * 521.0 / Hm)
    if sol is None:
        print(f"{d:>5}{nn_e:+9.1f}{'失败':>8}")
        continue
    cx, cy, D_px, m, rms = sol
    D_m = D_px * PXEFF / (Hm / 521.0)
    fit_e = D_m * 1000 - gt
    nn_es.append(abs(nn_e)); fit_es.append(abs(fit_e))
    print(f"{d:>5}{nn_e:+9.1f}{fit_e:+8.1f}{m:>5}{rms:8.2f}{D_px:9.1f}")

print(f"\n平均|Δ|: NN={np.mean(nn_es):.1f} → ring_fit={np.mean(fit_es):.1f} mm"
      f" （最差 {max(fit_es):.1f}）")
