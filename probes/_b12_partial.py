# -*- coding: utf-8 -*-
"""_b12_partial.py:B12② 部分环多环共享拟合·合成判决 v2(2026-11-07)。

v1 教训(已入档):一阶模型 r=D·tan(2θ+a·cosφ+b·sinφ) 在 rot/2θ 大时
物理失效(v1 诊断:小角环残差 549px——2θ+rot·cosφ 过零,tan 发散)。
v2:合成与拟合**同用 pyFAI 正向 tth 场**(make_pyfai_fwd,与生产
fit_tilt 同模型),物理正确。

三级流水(合成判决,检测层只给"粗"输入模拟真实):
  合成:fwd 几何画环(D,中心,rot1 可配)+ 泊松;
  检测层:粗中心(偏 200px 模拟斑点质心估计)+ 1D 径向谱粗 r_j
        (有意保留部分弧+椭圆的选择偏差——1D 粗值就是歪的);
  精化层:每 (环 j,方位 φ) 沿径向射线抛物线精化 → 2D 控制点;
  拟合层:残差 = fwd(D,p1,p2,r1,r2,d1,d2) − tth_j,5 参全局 LSQ
        (D,中心,rot1,rot2 共享全部环——多环锁死 R_j)。

判决几何(分离两因子):
  G1 中心画布内 + rot1=−29° + 遮 40% 方位(Fairchild 主因:大倾角+部分弧)
  G2 中心出画布 −300px + rot1=−29° + 遮 40%(B11e 场景)
判据:|ΔD|<2mm、控制点拟合 rms<1.5px、粗解→精化收敛。
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from poni_out import make_pyfai_fwd
from torchfai.phases import phase_qs

PX = 172e-6
LAM = 0.972386
H, W = 2527, 2463
Q = phase_qs("CeO2")[:9]
TTH = 2 * np.arcsin(np.clip(Q * LAM / (4 * np.pi), 0, 0.999))
fwd = make_pyfai_fwd(PX, PX, None)

D1G, D2G = np.meshgrid(np.arange(W), np.arange(H))   # d2=x, d1=y


def synth(rot1_rad, cx, cy, D_mm, mask_frac=0.40, counts=3000.0, seed=7):
    rng = np.random.default_rng(seed)
    tthf = fwd(D_mm * 1e-3, (cy + 0.5) * PX, (cx + 0.5) * PX,
               rot1_rad, 0.0, D1G.astype(float), D2G.astype(float))
    img = np.zeros((H, W))
    sig = np.deg2rad(0.06)
    for tq in TTH:
        img += counts * np.exp(-0.5 * ((tthf - tq) / sig) ** 2)
    y, x = np.mgrid[0:H, 0:W]
    phi = np.arctan2(y - cy, x - cx)
    img[(phi > np.pi * (1 - mask_frac))] = 0.0     # 遮上侧 mask_frac
    return rng.poisson(img).astype(float)


def control_points(img, cx0, cy0, D0_px):
    """检测层(粗)+ 精化层:1D 粗 r_j(含选择偏差)→ 射线抛物线 2D 点。"""
    y, x = np.mgrid[0:H, 0:W]
    r = np.hypot(x - cx0, y - cy0)
    nb = np.arange(0, r.max() + 2, 2.0)
    prof, _ = np.histogram(r, bins=nb, weights=img)
    cnt, _ = np.histogram(r, bins=nb)
    prof = prof / np.maximum(cnt, 1)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(prof, prominence=0.05 * prof.max(), distance=10)
    rj = (pk + 0.5) * 2.0
    # 序指派:弧峰升序 ↔ 线 tan2θ 升序(合成 9 环全可见部分)
    order = np.argsort(rj)[:len(TTH)]
    rj = rj[order]
    tt_s = np.sort(TTH)
    # 精化:每环取可见方位 12 射线,±6px 抛物线
    yv, xv = np.mgrid[0:H, 0:W]
    phi = np.arctan2(yv - cy0, xv - cx0)
    vis = phi <= np.pi * (1 - 0.40)
    pts = []
    for j, r0 in enumerate(rj):
        phs = np.linspace(-np.pi * 0.9, np.pi * 0.55, 14)
        for ph in phs:
            rs = np.arange(max(10, r0 - 8), min(r.max() - 2, r0 + 8), 1.0)
            xi = cx0 + rs * np.cos(ph); yi = cy0 + rs * np.sin(ph)
            ok = (xi > 2) & (xi < W - 3) & (yi > 2) & (yi < H - 3)
            if ok.sum() < 7:
                continue
            v = img[yi[ok].astype(int), xi[ok].astype(int)]
            i = int(np.argmax(v))
            if i <= 0 or i >= len(v) - 1 or v[i] < 5:
                continue
            den = (v[i - 1] - 2 * v[i] + v[i + 1])
            dd = 0.5 * (v[i - 1] - v[i + 1]) / den if den != 0 else 0.0
            rc = rs[ok][i] + np.clip(dd, -1, 1)
            pts.append((cx0 + rc * np.cos(ph), cy0 + rc * np.sin(ph),
                        np.searchsorted(tt_s, TTH[j]) - 1))
    if not pts:
        return None, None, None
    d2 = np.array([p[0] for p in pts]) - 0.5
    d1 = np.array([p[1] for p in pts]) - 0.5
    jj = np.clip(np.array([p[2] for p in pts]), 0, len(TTH) - 1)
    # 指派修正:j 是 TTH 原序;tt_s 排序后序号映射回原序
    inv = np.argsort(TTH)          # inv[k] = 第 k 小的线的原 index
    jj = inv[jj]
    return d1, d2, jj


def fit5(d1, d2, jidx, p_init):
    tt = TTH

    def resid(p):
        return fwd(p[0], p[1], p[2], p[3], p[4], d1, d2) - tt[jidx]
    f = least_squares(resid, p_init, loss="soft_l1", f_scale=1e-3,
                      max_nfev=8000)
    return f


def trial(tag, rot1, cx, cy, D_mm):
    """合成判决·拟合层:控制点=真值+0.5px 噪声(指派给定),初值粗
    (D −10%,中心偏 +200/−150px,rot 减半)——判 5 参共享拟合的
    收敛域与可辨识性。检测层(L2 RANSAC)是真实数据判决的事。"""
    rng = np.random.default_rng(5)
    D_px = D_mm * 1e-3 / PX
    # 真值控制点:fwd 场上每环可见方位 14 射线,r 由 fwd 反解(数值)
    d1s, d2s, jjs = [], [], []
    phs = np.linspace(-np.pi * 0.9, np.pi * 0.5, 22)
    for j, tq in enumerate(TTH):
        for ph in phs:
            rs = np.linspace(30, 4200, 3000)
            xi = cx + rs * np.cos(ph); yi = cy + rs * np.sin(ph)
            ok = (xi > 2) & (xi < W - 3) & (yi > 2) & (yi < H - 3)
            if ok.sum() < 100:
                continue
            tt = fwd(D_mm * 1e-3, (cy + 0.5) * PX, (cx + 0.5) * PX,
                     rot1, 0.0, yi[ok] + 0.0, xi[ok] + 0.0)
            k = int(np.argmin(np.abs(tt - tq)))
            if abs(tt[k] - tq) > 2e-3:
                continue
            rc = rs[ok][k] + rng.normal(0, 0.5)
            d2s.append(cx + rc * np.cos(ph) - 0.5)
            d1s.append(cy + rc * np.sin(ph) - 0.5)
            jjs.append(j)
    d1 = np.array(d1s); d2 = np.array(d2s); jidx = np.array(jjs)
    if len(d1) < 30:
        print(f"\n[{tag}] 控制点不足({len(d1)})")
        return
    # 粗初值:D −10%,中心 (+200,−150),rot 减半,rot2=0
    cx0, cy0 = cx + 200.0, cy - 150.0
    p0 = [D_px * PX * 0.90, (cy0 + 0.5) * PX, (cx0 + 0.5) * PX,
          rot1 * 0.5, 0.0]
    f = fit5(d1, d2, jidx, p0)
    Dm = f.x[0] * 1e3
    rr = np.hypot(d1 - f.x[1] / PX + 0.5, d2 - f.x[2] / PX + 0.5)
    rms_px = float(np.sqrt(np.mean((f.fun * rr / np.cos(TTH[jidx]) ** 2) ** 2)))
    ok = abs(Dm - D_mm) < 2.0
    print(f"\n[{tag}] 控制点 {len(d1)}(初值偏:中心(200,−150)px D−10%"
          f" rot×0.5)")
    print(f"  D={Dm:.2f}mm (GT {D_mm}, Δ{Dm-D_mm:+.3f}) "
          f"rot1={np.degrees(f.x[3]):+.2f}°(真 {np.degrees(rot1):+.2f}°)")
    print(f"  中心=({f.x[2]/PX-0.5:.0f},{f.x[1]/PX-0.5:.0f})"
          f"(真 ({cx:.0f},{cy:.0f})) rms={rms_px:.2f}px")
    print(f"  判定:{'✓ 通过' if ok and rms_px < 1.5 else '✗ 未过'}")


print("=" * 70)
print("B12② v2(物理 fwd 场):多环共享 5 参拟合,粗中心偏 200px,"
      "粗 D −10%,遮 40% 方位")
print("=" * 70)
trial("G1 中心在画布内 rot=−29°", np.deg2rad(-29.0), 1250.0, 1260.0, 200.19)
trial("G2 中心出画布−300px rot=−29°", np.deg2rad(-29.0), -300.0, 1263.0, 200.19)
