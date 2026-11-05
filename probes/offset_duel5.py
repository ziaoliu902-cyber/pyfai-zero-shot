# -*- coding: utf-8 -*-
"""offset_duel5.py：五臂离轴对决 + 种子归属归因（B 路线 Phase 4c 终版）。

臂：
- A 纯NN    : v4s1 六参直出
- B 几何中心 : 单种子经典
- C NN单种子 : NN 环心+盆域门
- E 经典级联 : {几何中心, RS定心, RS射线外推×4} 多种子 + 质量门择优（无NN）
- F 全系统   : E 候选 + NN 种子，质量门择优 → 归因胜者种子

RS=径向对称闭式定心（梯度线加权LSQ交点）。已知系统偏差：交点被拉向
像素质心 → 偏轴幅度收缩（GT 2×画布外时 RS 只走 1/3~1/2 路程），
方向正确 → 射线外推种子补幅度。

用法: python offset_duel5.py [n_per_bin]
"""
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import torch

from real_refine_v6 import load_px_nn
from solver_core import _r_model, consensus_pick, solve, ring_radii
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.phases import PHASES, PHASE_NAMES, phase_qs

N_PER_BIN = int(sys.argv[1]) if len(sys.argv) > 1 else 100
BINS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.45), (0.45, 2.5)]
PX = 344e-6


def bc(d, p1, p2, r1, r2):
    v2 = -d * np.tan(r1)
    v1 = np.tan(r2) * d / np.cos(r1)
    return (p2 + v2) / PX - 0.5, (p1 + v1) / PX - 0.5


# 公式自检：该点 tth 必须为 0
from poni_out import tth_of as _tth
_r = np.random.default_rng(0)
_w = 0.0
for _ in range(200):
    _d = 10 ** _r.uniform(-1, np.log10(3.0))
    _p1, _p2 = _r.uniform(0, 0.2, 2)
    _r1, _r2 = _r.uniform(-0.07, 0.07, 2)
    _cx, _cy = bc(_d, _p1, _p2, _r1, _r2)
    _w = max(_w, abs(float(_tth((_cy + 0.5) * PX, (_cx + 0.5) * PX, _d,
                                _p1, _p2, _r1, _r2))))
assert _w < 1e-9, f"环心公式误差 {_w:.1e}"


def rs_center(img):
    """径向对称定心（bgsub+σ3+pct85+鲁棒两遍）。收缩偏置已知，方向可靠。"""
    H, W = img.shape
    sm0 = gaussian_filter(img, 25.0)
    sm = gaussian_filter(img - sm0, 3.0)
    gx = ((sm[1:-1, 2:] - sm[1:-1, :-2]) * 0.5).ravel()
    gy = ((sm[2:, 1:-1] - sm[:-2, 1:-1]) * 0.5).ravel()
    yy, xx = np.mgrid[1:H - 1, 1:W - 1]
    xx = xx.ravel().astype(np.float64)
    yy = yy.ravel().astype(np.float64)
    g = np.hypot(gx, gy)
    pos = sm[1:-1, 1:-1].ravel()
    m = (g >= np.percentile(g, 85)) & (pos > 0)
    if m.sum() < 50:
        m = g >= np.percentile(g, 50)
    a, b, c = gy[m], -gx[m], gy[m] * xx[m] - gx[m] * yy[m]
    w = np.minimum(g[m], np.percentile(g[m], 99)) ** 2

    def solve_w(w):
        A11 = np.sum(w * a * a)
        A12 = np.sum(w * a * b)
        A22 = np.sum(w * b * b)
        B1 = np.sum(w * a * c)
        B2 = np.sum(w * b * c)
        det = A11 * A22 - A12 * A12
        if not np.isfinite(det) or abs(det) < 1e-12 * max(A11 * A22, 1e-30):
            return None
        return ((A22 * B1 - A12 * B2) / det,
                (A11 * B2 - A12 * B1) / det)

    o = solve_w(w)
    if o is None:
        return (W - 1) / 2.0, (H - 1) / 2.0
    r = a * o[0] + b * o[1] - c
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    keep = np.abs(r - np.median(r)) < 2.5 * s
    if keep.sum() > 50:
        o2 = solve_w(w * keep)
        if o2 is not None:
            o = o2
    return float(o[0]), float(o[1])


ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v4w3"), "val")
consts = ds.norm_consts
Hm, Wm = ds._mimgs.shape[1], ds._mimgs.shape[2]
m1, dev, _ = load_px_nn(r"runs\v4s1\best.pt")
m1.eval()

# ---- 选样：现实离轴子集（同 offset_duel）----
picks = {b: [] for b in BINS}
rng = np.random.default_rng(7)
order = rng.permutation(len(ds))
for i in order:
    g = int(ds._sel[i])
    p = ds._mparams[g]
    aux = ds._maux[g]
    if aux[1] < 4:
        continue
    if (abs(p[3]) > 0.02 or abs(p[4]) > 0.02
            or not (0.8 <= aux[5] <= 1.6)):
        continue
    ph = PHASE_NAMES[int(aux[0])]
    Q0 = phase_qs(ph)
    IR0 = PHASES[ph][1]
    k0 = IR0 >= 0.10
    rm0 = _r_model(Q0[k0], float(aux[5]), float(p[0]) / PX)
    if ((rm0 > 25) & (rm0 < min(Hm, Wm) * 0.707)).sum() < 5:
        continue
    cx_gt, cy_gt = bc(p[0], p[1], p[2], p[3], p[4])
    off = max(abs(cx_gt - (Wm - 1) / 2) / Wm,
              abs(cy_gt - (Hm - 1) / 2) / Hm)
    for b in BINS:
        if b[0] <= off < b[1] and len(picks[b]) < N_PER_BIN:
            picks[b].append(i)
            break
    if all(len(v) >= N_PER_BIN for v in picks.values()):
        break

print(f"== 五臂离轴对决[现实子集]："
      f"{sum(len(v) for v in picks.values())} 样本 ==")
print(f"{'箱':>10}{'n':>4} | {'A':>11}{'B':>11}{'C':>11}{'E级联':>11}"
      f"{'F全系统':>11} | F胜者种子")
HDR = f"{'':>10}{'':>4} | {'成功/中位':>11}" * 1


def solve_one(img, cx0, cy0, D_px0, Q, IR, lamA, D_ref):
    nb = max(256, int(1.05 * float(np.hypot(*img.shape)) / 2))
    peaks, prof, bin_px, good = ring_radii(img, cx0, cy0, nb=nb)
    return solve(peaks, cx0, cy0, D_px0, Q, lamA, PX, I_rel=IR,
                 canvas=img, prof_ctx=(prof, bin_px, good), D_ref_m=D_ref,
                 d_range=(0.05, 3.0) if D_ref == 0 else (0.35, 2.8))


def pick_best(sols):
    """质量门择优：环数↓ pscore↓ rms↑（同 solve_auto 优先级哲学）。"""
    ok = [s for s in sols if s is not None and s[3] >= 3]
    if not ok:
        return None
    return sorted(ok, key=lambda s: (-s[3], -s[6], s[4]))[0]


attr_tot = {}
attrG = {}
for b in BINS:
    idxs = picks[b]
    if not idxs:
        continue
    r = {k: [] for k in "ABCEFG"}
    t0 = time.time()
    for i in idxs:
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

        x = torch.from_numpy(
            np.log1p(img[None, None].astype(np.float32))).to(dev)
        with torch.no_grad():
            mu, _ = m1(x, torch.tensor([lamA], device=dev),
                       torch.tensor([1.0], device=dev))
        p6 = norm_to_param(mu.double().cpu().numpy(), consts, 1.0)[0]
        cxN, cyN = bc(p6[0], p6[1], p6[2], p6[3], p6[4])
        r["A"].append(abs(p6[0] - D_gt) / D_gt * 100)

        geo = ((Wm - 1) / 2, (Hm - 1) / 2)
        # B
        s = solve_one(img, *geo, 1.0 / PX, Q, IR, lamA, 0.0)
        r["B"].append(np.nan if (s is None or s[3] < 3)
                      else abs(s[2] * PX - D_gt) / D_gt * 100)
        # C
        s = solve_one(img, cxN, cyN, p6[0] / PX, Q, IR, lamA, float(p6[0]))
        r["C"].append(np.nan if (s is None or s[3] < 3)
                      else abs(s[2] * PX - D_gt) / D_gt * 100)
        # E：经典级联（几何 + RS + 射线外推）
        rsx, rsy = rs_center(img)
        gx0, gy0 = geo
        seeds = [("geo", gx0, gy0), ("rs", rsx, rsy)]
        for t in (1.0, 2.0, 3.5):
            seeds.append((f"ray{t}", rsx + t * (rsx - gx0),
                          rsy + t * (rsy - gy0)))
        sols = [solve_one(img, sx, sy, 1.0 / PX, Q, IR, lamA, 0.0)
                for _, sx, sy in seeds]
        bestE = pick_best(sols)
        r["E"].append(np.nan if bestE is None
                      else abs(bestE[2] * PX - D_gt) / D_gt * 100)
        # F：E 候选 + NN 种子，归因胜者
        sN = solve_one(img, cxN, cyN, p6[0] / PX, Q, IR, lamA, float(p6[0]))
        ok = [(nm, s) for (nm, sx, sy), s in zip(seeds, sols)
              if s is not None and s[3] >= 3]
        if sN is not None and sN[3] >= 3:
            ok.append(("nn", sN))
        if ok:
            nm, bestF = sorted(ok, key=lambda t: (-t[1][3], -t[1][6],
                                                  t[1][4]))[0]
            err_pct = abs(bestF[2] * PX - D_gt) / D_gt * 100
            r["F"].append(err_pct)
            if err_pct < 1:             # 与成功率同口径（<1%，非<100%）
                attr_tot[nm] = attr_tot.get(nm, 0) + 1
        else:
            r["F"].append(np.nan)
        # G：共识门（同一候选池，票数↓→pscore↓→m↓→rms↑）
        cp = consensus_pick(ok)
        if cp:
            nmG, bestG, _v = cp
            r["G"].append(abs(bestG[2] * PX - D_gt) / D_gt * 100)
            if abs(bestG[2] * PX - D_gt) / D_gt * 100 < 1:
                attrG[nmG] = attrG.get(nmG, 0) + 1
        else:
            r["G"].append(np.nan)

    row = f"{b[0]:.2f}~{b[1]:.2f}{len(idxs):>4} |"
    for k in "ABCEFG":
        v = np.array(r[k])
        row += (f"{np.mean(v < 1)*100:4.0f}%/"
                f"{np.nanmedian(v):6.1f}%")
    print(row + f"   [{time.time()-t0:.0f}s]")
print("\n(成功=|ΔD|/D<1% 且环≥3；A=NN直出 B=几何 C=NN单种子 "
      "E=经典级联 F=E+NN(m-first门) G=同池共识门)")
print("F 全域<1% 胜者种子归因:", dict(sorted(attr_tot.items(),
                                        key=lambda t: -t[1])))
print("G 全域<1% 胜者种子归因:", dict(sorted(attrG.items(),
                                        key=lambda t: -t[1])))
