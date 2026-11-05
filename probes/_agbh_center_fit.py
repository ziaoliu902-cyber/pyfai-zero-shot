# -*- coding: utf-8 -*-
"""AgBh 直测环心修复 → 结案：质心类方法在此帧类不可用（验尸存档）。

10-30 完整验尸链（全部实测）：
  1. 饱和平顶假设推翻：max 层仅 1px（260209），无饱和 blob。
  2. 四种独立估计器（top-0.1% 均值质心 / 剖面锐度网格 / 两级锁环
     Kasa / 2 号环区 SNR 搜索）全部收敛到 (263±5, 180±5)——因为
     挡光晕 r=40~120 与 r=120~280 的不对称亮台地（8 扇区亮度
     [191,3409,681,523,3444,143,78,77]，择优取向弧对）就在那里。
  3. 折叠对决（_agbh_fold_duel.py）：以 GT (194.3,239.7) 折叠
     ΣSNR=878，以直测 (263.6,179) 折叠 ΣSNR=455（1.93×），且
     高阶环 4/5 SNR 256/257 vs 63/64——环真实以 GT 为心。
     → 参考 poni 束心无罪；λ 元数据 +3.2% 仍是唯一 GT 缺陷。
  4. 推论：自由 poni 精修的 -2998.9 盆地（poni≈264,180）是 nuisance
     陷阱（径向背景模型把晕台地拟合进去，赚 24 nats）；dist 结论
     +47~+50mm 在两种中心下一致（乘积线，对此整个争端鲁棒）。

工程结论：光环帧的 poni 初始化只能靠 NN 头 / 画布中心 + poni 钉死
只修 dist（final 协议已是如此，无需改动）；ring_centroid 降级为
"晕团定位器/诊断"，不得作为 poni 来源。
"""
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from _agbh_final import ring_centroid, PX

REAL = os.path.join(ROOT, "data", "real")


def kasa_center(yy, xx):
    """Kasa 代数圆拟合：min Σ(x²+y²+a x+b y+c)² 的闭式解，返回 (cy,cx,r)。"""
    A = np.column_stack([xx, yy, np.ones_like(xx)])
    b = xx * xx + yy * yy
    (a, b2, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = a / 2, b2 / 2
    r = np.sqrt(max(c + cx * cx + cy * cy, 0.0))
    return cy, cx, r


def _profile_score(rpix, wpix, bins):
    """径向剖面锐度：平滑直方图的 std/mean（环越聚焦在少数 bin 分越高）。"""
    h, _ = np.histogram(rpix, bins=bins, range=(0.0, bins[-1]), weights=wpix)
    k = np.array([0.25, 0.5, 0.25])
    hs = np.convolve(h, k, mode="same")
    hs = np.clip(hs, 0, None)
    return hs.std() / (hs.mean() + 1e-9)


def ring_center_profile(img, frac=0.005, span=100.0, coarse=10.0,
                        fine=2.0, nb=240, rmax_px=520.0):
    """剖面锐度最大化环心：粗网格 → 细网格两级搜索。

    对弧截断/多环/环强度不均匀鲁棒（只用 top-frac 亮像素的径向聚焦度）。
    返回 (cy, cx, score)。
    """
    H, W = img.shape
    th = np.quantile(img, 1 - frac)
    yy, xx = np.mgrid[0:H, 0:W]
    m = img >= th
    ys, xs, ws = yy[m], xx[m], img[m]
    cy, cx = ys.mean(), xs.mean()                 # 初值（旧质心）
    bins = np.linspace(0.0, rmax_px, nb + 1)
    for step in (coarse, fine):
        best, bc = -1.0, (cy, cx)
        for dy in np.arange(-span, span + 1e-9, step):
            for dx in np.arange(-span, span + 1e-9, step):
                r = np.hypot(ys - (cy + dy), xs - (cx + dx))
                sel = r < rmax_px
                if sel.sum() < 100:
                    continue
                s = _profile_score(r[sel], ws[sel], bins)
                if s > best:
                    best, bc = s, (cy + dy, cx + dx)
        cy, cx = bc
        span = step                        # 细层搜索半径=粗层步长
    return cy, cx, best


def ring_center_fit(img, frac=0.001, rmin=30.0, rmax=220.0, iters=4,
                    min_px=80):
    """迭代环带 + 圆拟合环心（替换旧版均值质心）。"""
    H, W = img.shape
    th = np.quantile(img, 1 - frac)
    yy, xx = np.mgrid[0:H, 0:W]
    m = img >= th
    cy, cx = yy[m].mean(), xx[m].mean()          # 仅作初值
    for _ in range(iters):
        r = np.hypot(yy - cy, xx - cx)
        m2 = m & (r > rmin) & (r < rmax)
        if m2.sum() < min_px:
            return cy, cx, int(m2.sum()), False  # 退化：回退均值
        cy, cx, rf = kasa_center(yy[m2], xx[m2])
    return cy, cx, int(m2.sum()), True


def ring1_center(img, r_lo=100.0, r_hi=300.0, frac=0.05, band=22.0,
                 iters=6, init=None):
    """两级锁环环心：2 号环定位（亮团够不着）→ 1 号环抛光。

    病根修复（10-30 实测）：top-5% 亮像素 90% 挤在 r=60~120px 的
    挡光斑散射晕里——旧质心/剖面锐度全被亮团劫持（偏 ~86px），
    而从远初值锁 1 号环又会把亮团误当环。故先在 [230,400] 高半径
    区锁 2 号环定心（弧≥180°，圆拟合稳），再转 [120,210] 抛光。
    返回 (cy, cx, r_lock, n_used)。
    """
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    th = np.quantile(img, 1 - frac)
    m0 = img >= th
    cy, cx = init if init is not None else (H / 2, W / 2)

    def lock(lo, hi, bw, c0, n_it, keep=0.2):
        cy_, cx_ = c0
        r1, n = float("nan"), 0
        for _ in range(n_it):
            r = np.hypot(yy - cy_, xx - cx_)
            rb = np.arange(0.0, 640.0, 4.0)
            prof, _ = np.histogram(r[m0], bins=rb, weights=img[m0])
            sel = (rb[:-1] >= lo) & (rb[:-1] <= hi)
            k = int(np.argmax(np.where(sel, prof, -np.inf)))
            r1 = rb[k] + 2.0
            m = m0 & (np.abs(r - r1) < bw)
            if m.sum() < 60:
                break
            # 带内提纯：只留最亮 keep 比例（环脊线），剔除背景杂散
            th_b = np.quantile(img[m], 1 - keep)
            m = m & (img >= th_b)
            if m.sum() < 40:
                break
            cy_, cx_, _ = kasa_center(yy[m], xx[m])
            n = int(m.sum())
        return cy_, cx_, r1, n

    cy, cx, r2, n2 = lock(230.0, 400.0, 24.0, (cy, cx), 5)
    cy, cx, r1, n1 = lock(120.0, 210.0, 16.0, (cy, cx), 4)
    return cy, cx, r1, n1


def ring_center_snr(img, init=None, frac=0.003):
    """SNR 环心搜索：以 2 号环半径区 [280,380] 的折叠聚焦度为目标。

    设计依据（10-30 验尸）：挡光晕团 r≤120px、环 1 区被其污染；
    2 号环区任何近心候选下亮团都够不着 → 天然免疫。两级网格
    （步长 12→3px）后用 Kasa 圆拟合在 2 号环带上抛光。
    返回 (cy, cx, r2, n)。
    """
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    th = np.quantile(img, 1 - frac)
    m0 = img >= th
    ys, xs, ws = yy[m0], xx[m0], img[m0]
    cy, cx = init if init is not None else (H / 2, W / 2)
    bins2 = np.arange(280.0, 384.0, 4.0)

    def ring2_snr(c):
        r = np.hypot(ys - c[0], xs - c[1])
        h, _ = np.histogram(r, bins=bins2, weights=ws)
        base = h.mean()
        return h.max() / (base + 1e-9) if base > 0 else 0.0

    span = 96.0
    for step in (12.0, 3.0):
        best, bc = -1.0, (cy, cx)
        for dy in np.arange(-span, span + 1e-9, step):
            for dx in np.arange(-span, span + 1e-9, step):
                c = (cy + dy, cx + dx)
                s = ring2_snr(c)
                if s > best:
                    best, bc = s, c
        cy, cx = bc
        span = step
    # 2 号环带 Kasa 抛光
    r = np.hypot(ys - cy, xs - cx)
    h, e = np.histogram(r, bins=bins2, weights=ws)
    r2 = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    band = np.abs(r - r2) < 14.0
    if band.sum() >= 60:
        cy, cx, _ = kasa_center(ys[band], xs[band])
    return cy, cx, r2, int(band.sum())


if __name__ == "__main__":
    import time
    img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
    img = img.astype(np.float64)
    import pyFAI
    p = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))
    gy, gx = p.poni1 / PX, p.poni2 / PX

    cy0, cx0, n0 = ring_centroid(img)
    e0 = np.hypot(cy0 - gy, cx0 - gx)
    print(f"GT 环心 px = ({gy:.1f},{gx:.1f})")
    print(f"旧·均值质心  ({cy0:.1f},{cx0:.1f})  Δ=({cy0-gy:+.1f},"
          f"{cx0-gx:+.1f})  |Δ|={e0:.1f}px  n={n0}")

    t0 = time.time()
    cy2, cx2, sc = ring_center_profile(img)
    e2 = np.hypot(cy2 - gy, cx2 - gx)
    dt = time.time() - t0
    print(f"新·剖面锐度  ({cy2:.1f},{cx2:.1f})  Δ=({cy2-gy:+.1f},"
          f"{cx2-gx:+.1f})  |Δ|={e2:.1f}px  score={sc:.2f}  {dt:.1f}s")

    t0 = time.time()
    cy3, cx3, r1, n3 = ring1_center(img, init=(cy0, cx0))
    e3 = np.hypot(cy3 - gy, cx3 - gx)
    dt = time.time() - t0
    print(f"新·两级锁环  ({cy3:.1f},{cx3:.1f})  Δ=({cy3-gy:+.1f},"
          f"{cx3-gx:+.1f})  |Δ|={e3:.1f}px  r1={r1:.0f}  n={n3}  {dt:.1f}s")
    ok3 = e3 < 15
    print(f"\n结论：{'锁环法达标，合入' if ok3 else '锁环法未达标，需排查'}")

    t0 = time.time()
    cy4, cx4, r2, n4 = ring_center_snr(img)
    e4 = np.hypot(cy4 - gy, cx4 - gx)
    dt = time.time() - t0
    print(f"新·SNR搜心    ({cy4:.1f},{cx4:.1f})  Δ=({cy4-gy:+.1f},"
          f"{cx4-gx:+.1f})  |Δ|={e4:.1f}px  r2={r2:.0f}  n={n4}  {dt:.1f}s")
    print(f"\n终判：{'SNR 搜心达标' if e4 < 15 else 'SNR 搜心未达标'}")
