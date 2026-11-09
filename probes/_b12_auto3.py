# -*- coding: utf-8 -*-
"""_b12_auto3.py:B12② 自动指派 v6——正交分解:锐度对齐 + 缩放匹配
(2026-11-08)。

v4/v5 失败机理统一:fold 密度和耦合了「几何对齐」(中心/rot)与
「标度匹配」(D↔线表)——宽容差下糊谱骗分,紧容差下自洽子集局部
极小锁死(v5 合成 D=153=初值未动)。

v6 正交分解:
  ① **锐度对齐**(与 D、与线表全无关):score_sharp(p)=谱能量集中度
     Σ(h/max)² 或 Σh²/L0——斑点 tth 分层越锐越高。错误 rot/中心→
     环糊→锐度低,天然鉴别;D 只平移环位,不影响锐度 → 网格只搜
     中心×rot;
  ② 对齐几何下折叠谱 → find_peaks 提环位 {t_k}(无监督,不认线表);
  ③ **缩放匹配**:每对 (k,j) 假设 s=t_j/t_k → 贪心计数
     |s·t_i − t_m|<ε 匹配 → 最优 s 定 D=D_grid×s(线表只提供间距
     模式,序对齐此时干净);
  ④ 内点指派 + fit5(对齐已好,ICP 一宽一紧)。
判决:合成 G1/G2(rot 未知+中心 350px)+ Fairchild 两臂。
"""
import os
import sys

import numpy as np
import fabio
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from poni_out import make_pyfai_fwd
from torchfai.phases import phase_qs

PX_ = 172e-6


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
    """v6.1:返回 (x, y, 强度)——强度加权抑制噪声点海的伪锐度。"""
    H, W = img.shape
    bg = uniform_filter(img, 61)
    resid = img - bg
    v = resid.ravel()
    mu = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - mu))) + 1e-12
    mx = maximum_filter(img, size=5)
    ys, xs = np.nonzero((img >= mx) & (resid > ksig * sig))
    sx, sy, sh = [], [], []
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 3), min(H, y + 4)
        x0, x1 = max(0, x - 3), min(W, x + 4)
        w = np.clip(resid[y0:y1, x0:x1], 0, None)
        if w.sum() <= 0:
            continue
        gy_, gx_ = np.mgrid[y0:y1, x0:x1]
        sx.append((w * gx_).sum() / w.sum())
        sy.append((w * gy_).sum() / w.sum())
        sh.append(resid[y, x])
    return np.array(sx), np.array(sy), np.array(sh)


def auto_v6(sx, sy, sh, px, lam, qs, D0):
    """v6.1:D0=粗 D(mm);中心/rot 全靠强度加权锐度网格搜。"""
    global PX_
    PX_ = px
    fwd = make_pyfai_fwd(px, px, None)
    tth_q = 2 * np.arcsin(np.clip(qs * lam / (4 * np.pi), 0, 0.9999))
    d1, d2 = sy, sx
    wt = np.clip(sh - np.median(sh), 1.0, None)     # 强度权重(去底)
    H_est = max(d1.max(), 1)
    W_est = max(d2.max(), 1)
    tlo = 0.03
    edges = np.arange(0, tth_q[-1] * 1.2, 2e-4)
    ker = np.exp(-0.5 * (np.arange(-30, 31) / 10.0) ** 2)

    def hist(p):
        tm = fwd(p[0], p[1], p[2], p[3], p[4], d1, d2)
        okw = (tm > tlo) & (tm < edges[-1])
        h, _ = np.histogram(tm[okw], bins=edges, weights=wt[okw])
        return np.convolve(h, ker, mode="same").astype(float)

    # ① 锐度对齐:中心(斑点范围 ±600 步 50)× rot(步 10°)×D 粗档
    rot_g = [0.0] + [np.deg2rad(x) for x in (10, 20, 30, 40)]
    rot_g += [-r for r in rot_g[1:]]
    Dg = D0 * 1e-3
    best = None
    for cx in np.arange(max(0, W_est * 0.5 - 600), W_est + 600, 50):
        for cy in np.arange(max(0, H_est * 0.5 - 600), H_est + 600, 50):
            for r1 in rot_g:
                h = hist([Dg, (cy + 0.5) * px, (cx + 0.5) * px, r1, 0.0])
                if h.sum() < 30:
                    continue
                sharp = float((h ** 2).sum() / h.sum())   # 加权集中度
                if best is None or sharp > best[0]:
                    best = (sharp, cx, cy, r1)
    if best is None:
        return None
    sharp, cx, cy, r1 = best
    # ② 对齐谱提环位
    h = hist([Dg, (cy + 0.5) * px, (cx + 0.5) * px, r1, 0.0])
    pk, _ = find_peaks(h, prominence=0.05 * h.max())
    t_k = (edges[pk] + edges[np.minimum(pk + 1, len(edges) - 1)]) / 2
    t_k = t_k[(t_k > 0.02) & (t_k < tth_q[-1] * 1.1)]
    if len(t_k) < 3:
        return None
    # ③ 缩放匹配:t_k×s ≈ tth_q 子序列
    best_m = None
    eps = 4e-4
    for ka in range(len(t_k)):
        for jb in range(len(tth_q)):
            s = tth_q[jb] / t_k[ka]
            if not (0.5 < s < 2.0):
                continue
            tt = t_k * s
            used_j, match = set(), 0
            for t in tt:
                dd = np.abs(tth_q - t)
                j = int(np.argmin(dd))
                if dd[j] < eps and j not in used_j:
                    used_j.add(j)
                    match += 1
            if best_m is None or match > best_m[0]:
                best_m = (match, s)
    if best_m is None or best_m[0] < 3:
        return None
    match, s = best_m
    D_fit = Dg * s
    # ④ 内点指派 + fit5(ICP 一宽一紧)
    p0 = [D_fit, (cy + 0.5) * px, (cx + 0.5) * px, r1, 0.0]
    rng_d = (D_fit * 0.5, D_fit * 2.0)
    out = icp(fwd, d1, d2, tth_q, list(p0), (0.15, 0.07), d_range=rng_d)
    if out[0] is None:
        return dict(sharp=sharp, cx=cx, cy=cy, rot1=np.degrees(r1),
                    t_k=len(t_k), match=match, D0=D_fit * 1e3, fit=None)
    (pf, n, rms) = out[0]
    return dict(sharp=sharp, cx=cx, cy=cy, rot1=np.degrees(r1), t_k=len(t_k),
                match=match, D0=D_fit * 1e3, fit=(pf, n, rms))


# ================= 判决 =================
PX = 172e-6
LAM = 0.972386
H, W = 2527, 2463
Q = phase_qs("CeO2")[:9]
TTH = 2 * np.arcsin(np.clip(Q * LAM / (4 * np.pi), 0, 0.999))
fwd_s = make_pyfai_fwd(PX, PX, None)
D1G, D2G = np.meshgrid(np.arange(W).astype(float),
                       np.arange(H).astype(float))

print("=" * 66)
print("自动指派 v6(锐度对齐+缩放匹配):4 案例判决(D0 粗给 ±10%)")
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
    sx, sy, sh_ = spots_of(simg)
    res = auto_v6(sx, sy, sh_, PX, LAM, Q, Dm * 0.9)
    print(f"\n[{tag}] 斑点 {len(sx)}")
    if res is None:
        print("  失败(锐度网格/环位/匹配)")
        continue
    print(f"  对齐:中心=({res['cx']:.0f},{res['cy']:.0f}) rot1="
          f"{res['rot1']:+.0f}°(真 ({cx:.0f},{cy:.0f}) −29°) "
          f"环位 {res['t_k']} 匹配 {res['match']} 线 D0={res['D0']:.1f}mm")
    if res["fit"] is None:
        print("  ICP 失败")
        continue
    pf, n, rms = res["fit"]
    Df = pf[0] * 1e3
    print(f"  终拟:内点 {n} rms {rms:.2f}px D={Df:.2f}mm "
          f"Δ{Df-Dm:+.3f} rot1={np.degrees(pf[3]):+.2f}° "
          f"中心=({pf[2]/PX-0.5:.0f},{pf[1]/PX-0.5:.0f}) "
          f"{'✓' if abs(Df-Dm)<2 else '✗'}")

print("\n" + "=" * 66)
PX = 15e-6
LAM = 1.540
D_GT = 88.2065
gt_cx = (0.0449458 / PX) - 0.5
gt_cy = (0.0302286 / PX) - 0.5
QS = phase_qs("Al2O3")
img = fabio.open(ROOT + r"\data\real\Fairchild.edf").data.astype(np.float64)
sx, sy, sh_ = spots_of(img)
for tag in ("Fairchild(D0=GT)",):
    res = auto_v6(sx, sy, sh_, PX, LAM, QS, D_GT)
    print(f"\n[{tag}] 斑点 {len(sx)}")
    if res is None:
        print("  失败")
        continue
    print(f"  对齐:中心=({res['cx']:.0f},{res['cy']:.0f}) rot1="
          f"{res['rot1']:+.0f}°(GT ({gt_cx:.0f},{gt_cy:.0f}) −29°) "
          f"环位 {res['t_k']} 匹配 {res['match']} 线 D0={res['D0']:.1f}mm")
    if res["fit"] is None:
        print("  ICP 失败")
        continue
    pf, n, rms = res["fit"]
    Df = pf[0] * 1e3
    print(f"  终拟:内点 {n} rms {rms:.2f}px D={Df:.2f}mm "
          f"Δ{Df-D_GT:+.2f} rot=({np.degrees(pf[3]):+.1f},"
          f"{np.degrees(pf[4]):+.1f})° "
          f"中心=({pf[2]/PX-0.5:.0f},{pf[1]/PX-0.5:.0f}) "
          f"{'✓<5mm' if abs(Df-D_GT)<5 else '✗'}")
