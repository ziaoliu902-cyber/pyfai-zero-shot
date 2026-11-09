# -*- coding: utf-8 -*-
"""_b12_auto.py:B12② 自动指派 v2——假设网格 + ICP 细化(2026-11-07)。

v1 教训:rot=0 初值在大倾角域=坏盆地(rot–D 简并陷阱,D 180mm 吸收
倾角,ICP 指派一致率 18% 恶性循环;Fairchild 崩到 D≈1mm)。
v2:**rot1×D 网格假设** × ICP 细化:
  rot1 ∈ {0,±10°,±20°,±30°,±40°} × D ∈ {0.8,0.9,1.0,1.1}×粗
  → 每组 ICP 2 轮(tol=0.10 gap)→ 仲裁 = 指派内点数(多者优先,
  同数比 rms)→ 最优组 ICP 3 轮终拟。
判决:合成 G1/G2 指派一致率+Δ;Fairchild 自动全链 Δ<5mm。
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
GRID_D = [0.8, 0.9, 1.0, 1.1]


def icp(fwd, d1, d2, tth_q, p, tols, min_n=30, d_range=(10e-3, 2.0)):
    """退火 ICP:tol 逐轮收紧(tols 如 (0.30,0.20,0.10))。
    bounds 防 D→∞ 退化解(rms→0 内点=全部的伪最优)。"""
    m = None
    lo = [d_range[0], p[1] - 1000 * px_, p[2] - 1000 * px_,
          np.deg2rad(-60), np.deg2rad(-60)]
    hi = [d_range[1], p[1] + 1000 * px_, p[2] + 1000 * px_,
          np.deg2rad(60), np.deg2rad(60)]
    p = np.clip(np.array(p, float), lo, hi)
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
    rr = np.hypot(a - p[1] / px_ + 0.5, b - p[2] / px_ + 0.5)
    rms = float(np.sqrt(np.mean((f.fun * rr / np.cos(tth_q[j]) ** 2) ** 2)))
    return (p, int(m.sum()), rms), int(m.sum()), rms


def fold_grid_icp(sx, sy, px, lam, qs, cx0, cy0, D0):
    """v3:谱折叠网格 + ICP 细化。

    每个假设(中心格×rot 格×D 档)下把斑点 tth 折叠成 1D 直方图,
    score=Σ_j 谱密度(线 tth_j 处)——伪解需要多条线同时有真浓度,
    噪声点海难凑分。最优格 → ICP 细化(中心初值=格点)。
    """
    fwd = make_pyfai_fwd(px, px, None)
    tth_q = 2 * np.arcsin(np.clip(qs * lam / (4 * np.pi), 0, 0.9999))
    d1, d2 = sy, sx
    rng_d = (D0 * 1e-3 * 0.3, D0 * 1e-3 * 3.0)
    tmax = tth_q[-1] * 1.15
    edges = np.arange(0, tmax, 2e-4)
    ker = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)

    def score(p):
        tm = fwd(p[0], p[1], p[2], p[3], p[4], d1, d2)
        h, _ = np.histogram(tm, bins=edges)
        h = np.convolve(h, ker, mode="same")
        # 线位置处谱密度(线性插值)+ 线间谷底惩罚
        idx = np.clip(np.searchsorted(edges, tth_q) - 1, 0, len(h) - 1)
        return float(h[idx].sum())

    best = None
    for dx in (-300, -150, 0, 150, 300):
        for dy in (-300, -150, 0, 150, 300):
            for r1 in list(GRID_R) + list(-GRID_R[1:]):
                for fd in (0.85, 1.0, 1.15):
                    p0 = [D0 * fd * 1e-3, (cy0 + dy + 0.5) * px,
                          (cx0 + dx + 0.5) * px, r1, 0.0]
                    s = score(p0)
                    if best is None or s > best[0]:
                        best = (s, p0)
    s, p0 = best
    # 阶段2:固定 rot/D 档(top 附近),中心 ±225px 步 25px 细扫
    best2 = None
    r1s = sorted(set([p0[3]] + [p0[3] - np.deg2rad(10),
                                p0[3] + np.deg2rad(10)]))
    fds = sorted(set([0.85, 1.0, 1.15] + [p0[0] / (D0 * 1e-3)]))
    cxg = p0[2] / px - 0.5
    cyg = p0[1] / px - 0.5
    for r1 in r1s:
        for fd in fds:
            for ddx in range(-9, 10):
                for ddy in range(-9, 10):
                    q = [D0 * fd * 1e-3,
                         (cyg + ddy * 25 + 0.5) * px,
                         (cxg + ddx * 25 + 0.5) * px, r1, 0.0]
                    s2 = score(q)
                    if best2 is None or s2 > best2[0]:
                        best2 = (s2, q)
    if best2[0] >= s:
        s, p0 = best2
    # ICP 细化(退火容差);失败退回粗格并标注
    out = icp(fwd, d1, d2, tth_q, list(p0), (0.25, 0.15, 0.08),
              d_range=rng_d)
    if out[0] is None:
        print(f"    [fold] score={s:.0f} 粗格 D={p0[0]*1e3:.1f}mm "
              f"rot1={np.degrees(p0[3]):.0f}° 中心=({p0[2]/px-0.5:.0f},"
              f"{p0[1]/px-0.5:.0f}) → ICP 细化失败(内点<30)")
        return None
    (pf, n, rms) = out[0]
    return (pf, n, rms, 0.0, 0), 0, 0


def grid_icp(sx, sy, px, lam, qs, cx0, cy0, D0):
    fwd = make_pyfai_fwd(px, px, None)
    tth_q = 2 * np.arcsin(np.clip(qs * lam / (4 * np.pi), 0, 0.9999))
    d1, d2 = sy, sx
    rng_d = (D0 * 1e-3 * 0.3, D0 * 1e-3 * 3.0)
    best = None
    for r1 in list(GRID_R) + list(-GRID_R[1:]):
        for fd in GRID_D:
            p0 = [D0 * fd, (cy0 + 0.5) * px, (cx0 + 0.5) * px, r1, 0.0]
            out = icp(fwd, d1, d2, tth_q, list(p0), (0.30, 0.20),
                      d_range=rng_d)
            if out[0] is None:
                continue
            (pf, n, rms) = out[0]
            if rms > 6.0 or n < 40:        # sanity:伪最优/弱假设不入池
                continue
            key = (n, -rms)
            if best is None or key > best[0]:
                best = (key, pf, n, rms, r1, fd)
    if best is None:
        return None
    _, p_best, n_b, rms_b, r1_b, fd_b = best
    out = icp(fwd, d1, d2, tth_q, list(p_best), (0.30, 0.20, 0.10),
              d_range=rng_d)
    if out[0] is None:
        return best[1] + (best[4], best[5])
    return out[0] + (r1_b, fd_b)


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


# ================= 合成 G1/G2 =================
PX = 172e-6
LAM = 0.972386
H, W = 2527, 2463
Q = phase_qs("CeO2")[:9]
TTH = 2 * np.arcsin(np.clip(Q * LAM / (4 * np.pi), 0, 0.999))
fwd_s = make_pyfai_fwd(PX, PX, None)
D1G, D2G = np.meshgrid(np.arange(W).astype(float),
                       np.arange(H).astype(float))
px_ = PX

print("=" * 66)
print("自动指派 v2(网格+ICP):合成 G1/G2(真 rot=−29°,D=200.19mm)")
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
    res = fold_grid_icp(sx, sy, PX, LAM, Q, cx + 200, cy - 150, Dm * 0.9)
    if res is None:
        print(f"\n[{tag}] 网格+ICP 全失败")
        continue
    p, n, rms = res[0][:3]
    print(f"\n[{tag}] 斑点 {len(sx)} → 内点 {n} rms {rms:.1f}px "
          f"(fold 网格粗选+ICP)")
    print(f"  D={p[0]*1e3:.2f}mm Δ{p[0]*1e3-Dm:+.3f}mm "
          f"rot1={np.degrees(p[3]):+.2f}°(真 −29.00) "
          f"中心=({p[2]/PX-0.5:.0f},{p[1]/PX-0.5:.0f})(真 ({cx:.0f},{cy:.0f})) "
          f"{'✓' if abs(p[0]*1e3-Dm) < 2 else '✗'}")

# ================= Fairchild =================
print("\n" + "=" * 66)
print("Fairchild 自动全链 v2")
PX = 15e-6
px_ = PX
LAM = 1.540
D_GT = 88.2065
gt_cx = (0.0449458 / PX) - 0.5
gt_cy = (0.0302286 / PX) - 0.5
QS = phase_qs("Al2O3")
img = fabio.open(ROOT + r"\data\real\Fairchild.edf").data.astype(np.float64)
sx, sy = spots_of(img)
for tag, dx, dy in (("中心偏(200,−150)", 200, -150),
                    ("中心偏(−200,150)", -200, 150)):
    res = fold_grid_icp(sx, sy, PX, LAM, QS, gt_cx + dx, gt_cy + dy, D_GT)
    if res is None:
        print(f"\n[{tag}] 全失败")
        continue
    p, n, rms = res[0][:3]
    print(f"\n[{tag}] 内点 {n} rms {rms:.1f}px "
          f"(fold 网格粗选+ICP)")
    print(f"  D={p[0]*1e3:.2f}mm Δ{p[0]*1e3-D_GT:+.2f}mm "
          f"rot=({np.degrees(p[3]):+.1f},{np.degrees(p[4]):+.1f})° "
          f"(GT −29.0,+0.96) "
          f"中心=({p[2]/PX-0.5:.0f},{p[1]/PX-0.5:.0f}) "
          f"(GT {gt_cx:.0f},{gt_cy:.0f}) "
          f"{'✓<5mm' if abs(p[0]*1e3-D_GT) < 5 else '✗'}")
