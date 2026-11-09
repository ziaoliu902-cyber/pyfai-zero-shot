# -*- coding: utf-8 -*-
"""_b12_auto2.py:B12② 自动指派 v5——对比度 fold + 紧容差多假设
仲裁(2026-11-08)。

v4 遗留:fold 密度和在错误 rot 格骗分(合成选 −10° 真 −29°;
Fairchild 臂2 选 −36°)。机理:宽容差/宽平滑下,错误 rot 把环拉糊,
糊能量仍覆盖线位,密度和不降反升。

v5 组合拳:
  ① fold score 改**对比度**:Σ_j (h(tth_j)−b)/b,b=谱背景中位——
    糊谱背景高、对比低,锐谱对比高;
  ② 多假设:粗格+中心细扫后取 **top5(rot 档去重)**,不押单格;
  ③ 每假设**紧容差 ICP** 退火 (0.12,0.08,0.05)——错误 rot 下环
    拉宽 >容差,内点饿死(<30 淘汰);正确 rot 下斑点 tth 集中
    ~1e-3 rad 存活;
  ④ 仲裁:紧容差内点数最大者胜,同数比 rms。
判决 4 案例:合成 G1/G2(rot 未知+中心 350px)、Fairchild 两臂。
"""
import os
import sys

import numpy as np
import fabio
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.optimize import least_squares

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from poni_out import make_pyfai_fwd
from torchfai.phases import phase_qs

GRID_R = np.deg2rad([0, 10, 20, 30, 40])
GRID_D = (0.85, 1.0, 1.15)
TOLS = (0.12, 0.08, 0.05)
PX_ = 172e-6            # 全局(中心 bounds 换算用),案例段更新


def icp(fwd, d1, d2, tth_q, p, tols, min_n=30, d_range=(10e-3, 2.0)):
    lo = [d_range[0], p[1] - 1000 * PX_, p[2] - 1000 * PX_,
          np.deg2rad(-60), np.deg2rad(-60)]
    hi = [d_range[1], p[1] + 1000 * PX_, p[2] + 1000 * PX_,
          np.deg2rad(60), np.deg2rad(60)]
    p = np.clip(np.array(p, float), lo, hi)
    m = None
    for tol in tols:
        tm = fwd(p[0], p[1], p[2], p[3], p[4], d1, d2)
        k = np.argmin(np.abs(tm[:, None] - tth_q[None, :]), axis=1)
        dtt = np.abs(tm - tth_q[k])
        gap = np.min(np.abs(tth_q[:, None] - tth_q[None, :])
                     + np.eye(len(tth_q)) * 9, axis=1)
        m = dtt < tol * gap[k]
        if m.sum() < min_n:
            return None, 0, np.inf
        a, b, j = d1[m], d2[m], k[m]

        def r_(pp):
            return fwd(pp[0], pp[1], pp[2], pp[3], pp[4], a, b) - tth_q[j]
        try:
            f = least_squares(r_, p, bounds=(lo, hi), max_nfev=8000)
        except Exception:
            return None, 0, np.inf
        p = f.x
    rr = np.hypot(a - p[1] / PX_ + 0.5, b - p[2] / PX_ + 0.5)
    rms = float(np.sqrt(np.mean((f.fun * rr / np.cos(tth_q[j]) ** 2) ** 2)))
    return (p, int(m.sum()), rms), int(m.sum()), rms


def spots_of(img, ksig=6.0):
    H, W = img.shape
    bg = uniform_filter(img, 61)
    resid = img - bg
    v = resid.ravel()
    mu = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - mu))) + 1e-12
    mx = maximum_filter(img, size=5)
    ys, xs = np.nonzero((img >= mx) & (resid > ksig * sig))
    sx, sy = [], []
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 3), min(H, y + 4)
        x0, x1 = max(0, x - 3), min(W, x + 4)
        w = np.clip(resid[y0:y1, x0:x1], 0, None)
        if w.sum() <= 0:
            continue
        gy_, gx_ = np.mgrid[y0:y1, x0:x1]
        sx.append((w * gx_).sum() / w.sum())
        sy.append((w * gy_).sum() / w.sum())
    return np.array(sx), np.array(sy)


def auto_v5(sx, sy, px, lam, qs, cx0, cy0, D0, n_top=5):
    """对比度 fold(top n_top 假设)+ 紧容差 ICP 多假设仲裁。"""
    global PX_
    PX_ = px
    fwd = make_pyfai_fwd(px, px, None)
    tth_q = 2 * np.arcsin(np.clip(qs * lam / (4 * np.pi), 0, 0.9999))
    d1, d2 = sy, sx
    rng_d = (D0 * 1e-3 * 0.3, D0 * 1e-3 * 3.0)
    edges = np.arange(0, tth_q[-1] * 1.15, 1e-4)
    ker = np.exp(-0.5 * (np.arange(-30, 31) / 10.0) ** 2)   # σ=1e-3 rad

    def fold(p):
        tm = fwd(p[0], p[1], p[2], p[3], p[4], d1, d2)
        h, _ = np.histogram(tm, bins=edges)
        h = np.convolve(h, ker, mode="same").astype(float)
        b = max(float(np.median(h[h > 0])), 1.0)
        idx = np.clip(np.searchsorted(edges, tth_q) - 1, 0, len(h) - 1)
        return float(np.sum(np.maximum(h[idx] - b, 0) / b))

    # 阶段1:粗格(中心 ±300 步 150)× rot × D
    cand = []
    for dx in (-300, -150, 0, 150, 300):
        for dy in (-300, -150, 0, 150, 300):
            for r1 in list(GRID_R) + list(-GRID_R[1:]):
                for fd in GRID_D:
                    p0 = [D0 * fd * 1e-3, (cy0 + dy + 0.5) * px,
                          (cx0 + dx + 0.5) * px, r1, 0.0]
                    cand.append((fold(p0), p0))
    cand.sort(key=lambda t: -t[0])
    # 阶段2:top 假设(rot 档去重)中心 ±225 步 25 细扫
    seen_r, hyps = [], []
    for s, p0 in cand:
        rr = round(float(np.degrees(p0[3])))
        if any(abs(rr - x) <= 5 for x in seen_r):
            continue
        seen_r.append(rr)
        cxg, cyg = p0[2] / px - 0.5, p0[1] / px - 0.5
        best = (s, p0)
        for ddx in range(-9, 10):
            for ddy in range(-9, 10):
                q = [p0[0], (cyg + ddy * 25 + 0.5) * px,
                     (cxg + ddx * 25 + 0.5) * px, p0[3], 0.0]
                s2 = fold(q)
                if s2 > best[0]:
                    best = (s2, q)
        hyps.append(best)
        if len(hyps) >= n_top:
            break
    # 阶段3:紧容差 ICP × 每假设,内点数仲裁
    results = []
    for s, q in hyps:
        out = icp(fwd, d1, d2, tth_q, list(q), TOLS, d_range=rng_d)
        if out[0] is None:
            continue
        (pf, n, rms) = out[0]
        results.append((n, -rms, pf, s, np.degrees(q[3])))
    if not results:
        return None, hyps
    results.sort(key=lambda t: (-t[0], t[1]))
    return results[0], hyps


# ================= 判决 =================
# ---- 合成 G1/G2 ----
PX = 172e-6
LAM = 0.972386
H, W = 2527, 2463
Q = phase_qs("CeO2")[:9]
TTH = 2 * np.arcsin(np.clip(Q * LAM / (4 * np.pi), 0, 0.999))
fwd_s = make_pyfai_fwd(PX, PX, None)
D1G, D2G = np.meshgrid(np.arange(W).astype(float),
                       np.arange(H).astype(float))

print("=" * 66)
print("自动指派 v5(对比度 fold + 紧容差多假设):4 案例判决")
for tag, cx, cy in (("G1 中心内", 1250.0, 1260.0),
                    ("G2 中心出画布", -300.0, 1263.0)):
    rot1 = np.deg2rad(-29.0)
    Dm = 200.19
    rng = np.random.default_rng(7)
    tthf = fwd_s(Dm * 1e-3, (cy + 0.5) * PX, (cx + 0.5) * PX,
                 rot1, 0.0, D1G, D2G)
    simg = np.zeros((H, W))
    for tq in TTH:
        simg += 3000 * np.exp(-0.5 * ((tthf - tq) / np.deg2rad(0.06)) ** 2)
    yv, xv = np.mgrid[0:H, 0:W]
    phi = np.arctan2(yv - cy, xv - cx)
    simg[phi > np.pi * 0.6] = 0.0
    simg = rng.poisson(simg).astype(float)
    sx, sy = spots_of(simg)
    res, hyps = auto_v5(sx, sy, PX, LAM, Q, cx + 200, cy - 150, Dm * 0.9)
    if res is None:
        print(f"\n[{tag}] 全假设饿死(紧容差内点<30)")
        continue
    n, neg_rms, p, s_fold, r_deg = res
    Df = p[0] * 1e3
    ok = abs(Df - Dm) < 2.0
    print(f"\n[{tag}] 斑点 {len(sx)} 假设 {len(hyps)} 档")
    print(f"  胜出:内点 {n} rms {-neg_rms:.2f}px fold对比度 {s_fold:.0f} "
          f"(粗格 rot {r_deg:.0f}°)")
    print(f"  D={Df:.2f}mm Δ{Df-Dm:+.3f} rot1={np.degrees(p[3]):+.2f}°"
          f"(真 −29.00) 中心=({p[2]/PX-0.5:.0f},{p[1]/PX-0.5:.0f})"
          f"(真 ({cx:.0f},{cy:.0f})) {'✓' if ok else '✗'}")

# ---- Fairchild ----
print("\n" + "=" * 66)
PX = 15e-6
LAM = 1.540
D_GT = 88.2065
gt_cx = (0.0449458 / PX) - 0.5
gt_cy = (0.0302286 / PX) - 0.5
QS = phase_qs("Al2O3")
img = fabio.open(ROOT + r"\data\real\Fairchild.edf").data.astype(np.float64)
sx, sy = spots_of(img)
for tag, dx, dy in (("中心偏(200,−150)", 200, -150),
                    ("中心偏(−200,150)", -200, 150)):
    res, hyps = auto_v5(sx, sy, PX, LAM, QS, gt_cx + dx, gt_cy + dy, D_GT)
    if res is None:
        print(f"\n[{tag}] 全假设饿死")
        continue
    n, neg_rms, p, s_fold, r_deg = res
    Df = p[0] * 1e3
    ok = abs(Df - D_GT) < 5.0
    print(f"\n[{tag}] 胜出:内点 {n} rms {-neg_rms:.2f}px fold对比度 "
          f"{s_fold:.0f}(粗格 rot {r_deg:.0f}°)")
    print(f"  D={Df:.2f}mm Δ{Df-D_GT:+.2f} "
          f"rot=({np.degrees(p[3]):+.1f},{np.degrees(p[4]):+.1f})° "
          f"(GT −29.0) 中心=({p[2]/PX-0.5:.0f},{p[1]/PX-0.5:.0f}) "
          f"(GT {gt_cx:.0f},{gt_cy:.0f}) {'✓<5mm' if ok else '✗'}")
