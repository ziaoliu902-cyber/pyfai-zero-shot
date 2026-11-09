# -*- coding: utf-8 -*-
"""solver_core.py：相库驱动的通用峰位几何解算器（B 路线核心库）。

2026-10-31 从 ring_fit.py（ceria 专用脚本）泛化：
- 相库驱动：q 表来自 torchfai.phases（CeO2/LaB6/Si/Al2O3/Quartz/AgBh/AgSt）
- 多相自动识别：逐相求解，按 环数↓ 票数↓ rms↑ 择优
- NN 初始化可选：无 NN 时中心取画布中心、D 全域投票（盆域门自动关闭）
- 画布无关：任意 (canvas, px_m) 组合

算法（ceria 七档验证：平均 2.8mm / 最差 6.7mm）：
径向均值谱找峰（抛物线内插）→ D 投票聚类（峰×hkl 隐含 D）
→ 比例容差迭代配对（3%→2%→1%，轮内快照最优）→ LSQ(cx,cy,D)
→ 全簇评估（环数↓ 票数↓ rms↑）+ NN 盆域门（±80mm，可选）。
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.phases import PHASE_NAMES, phase_qs

NB = 2048          # 径向谱 bin 数
MIN_R = 25.0       # 最小环半径(px)——切束流挡片
PROM = 0.05        # 峰显著性(×谱最大值)
BASIN_M = 0.08     # NN 盆域半宽(m)，NN 初始化才启用


def _lam_A(lam):
    """波长单位自防御：<1e-6 视为米 → 换 Å。"""
    return lam * 1e10 if lam < 1e-6 else float(lam)


def to_canvas(img, max_side=520):
    """整数倍面积重采样到 ≤max_side 的画布。
    返回 (canvas, px_m, cx0, cy0)；px_m=画布像素物理尺寸(m)，中心=几何中心。"""
    h, w = img.shape
    s = max(1, int(np.ceil(max(h, w) / max_side)))
    H, W = h // s, w // s
    canvas = img[:H * s, :W * s].reshape(H, s, W, s).mean(axis=(1, 3))
    return canvas.astype(np.float64), s, (W - 1) / 2.0, (H - 1) / 2.0


def undistort(img, sp, n_iter=2):
    """样条反畸变（T5，2026-11-02）：pyFAI 正向场 raw[i,j]=F(i+dy,j+dx)
    （源码实锤 p1=pixel1*(dY+d1c)）；理想网格图需解逆映射——不动点迭代
    src ← grid − disp(src) 两轮（二阶精度）。

    工程坑（全踩过）：
    · 正向场直接加 ≠ 校正（那是把 F 在位移点采样）——± 号经验判据会
      互矛盾，必须求逆；
    · pyFAI splineFuncX/Y 的 list_of_points=True 是"独立排序伪散点"，
      唯一值多则 N×N 网格爆炸；scipy 1.16 已删 bispeu——位移场在正则
      网格求一次（pyFAI grid 快路），散点采样用 map_coordinates 采场
      （场 ~100px 光滑，误差 <<0.001px）。
    验证：①合成往返 band|Δ| med 0.05px（_t5_roundtrip.py）；
    ②inv+平几何自检 ≡ raw+spline 自检（halfccd 0.66%/s0=0.3312 vs
    0.69%/0.3312，_t5_sharp.py/_t5_half2.py）。
    注意：零样本基准协议里求解器看不到 poni（spline 属 GT 侧）——
    本函数供用户给样条的辅助场景/未来接线。"""
    from scipy.ndimage import map_coordinates
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    DX = np.asarray(sp.splineFuncX(xx, yy), float)
    DY = np.asarray(sp.splineFuncY(xx, yy), float)
    src_y, src_x = yy.copy(), xx.copy()
    for _ in range(n_iter):
        ddy = map_coordinates(DY, [src_y, src_x], order=1, mode="nearest")
        ddx = map_coordinates(DX, [src_y, src_x], order=1, mode="nearest")
        src_y = yy - ddy
        src_x = xx - ddx
    return map_coordinates(img, [src_y, src_x], order=1, mode="nearest")


def concentric_center(img, c0, D_px, qs, lam, half=3.0, min_rings=2,
                      n_iter=4000):
    """共心约束中心真值（T9，2026-11-03）：所有环共用一个圆心、各自半径
    自由，让"像元到同一圆心的径向散布"最小的点——不依赖 GT/poni。

    判决依据（ceria 实证，_center_truth.py/_center_gt.py）：
    · σ 由目标函数曲率估出 0.02–0.03px，三档（150/250/300）互相一致
      到 ~1px（装置静态）——真中心极稳；
    · 而 GT poni 中心系统偏 15.5–17.9px：用 GT 中心提谱首环峰高只有
      我们中心的 1/3~1/5、宽度 18px vs 12–13px（GT 中心把环抹糊）。
    → 基准 Δcenter 应对照本函数（center_ref），GT 中心差单独成列归因。

    参数 c0/D_px/qs/lam 与 img 同坐标系（原生像素）：c0=(x0,y0) 初始
    中心（求解器解即可，收敛盆地 ~±20px），D_px 定环带位置。
    返回 dict(cx, cy, sx, sy, n, ok) 或 None（可用环 < min_rings）。"""
    from scipy.optimize import minimize
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    qs = np.asarray(qs, float)
    tt = 2.0 * np.arcsin(np.clip(float(lam) * qs / (4.0 * np.pi), 0, 0.999))
    radii = float(D_px) * np.tan(tt)
    r_edge = min(c0[0], c0[1], W - 1 - c0[0], H - 1 - c0[1])
    rings = []
    for r0 in radii:
        if r0 < 60 or r0 > 0.9 * r_edge:
            continue                      # 太小/不完整：不参评
        band = np.abs(np.hypot(xx - c0[0], yy - c0[1]) - r0) < half
        iy, ix = np.nonzero(band)
        if iy.size < 300:
            continue
        v = img[iy, ix]
        base = np.percentile(v, 40)
        keep = v > base + 0.5 * (np.percentile(v, 99) - base)
        if keep.sum() < 200:
            continue                      # 环弱/被吞：诚实拒收
        rings.append((ix[keep].astype(float), iy[keep].astype(float)))
    if len(rings) < min_rings:
        return None

    def _obj(c):
        tot = 0.0
        for px_, py_ in rings:
            d = np.hypot(px_ - c[0], py_ - c[1])
            tot += float(np.sum((d - d.mean()) ** 2))
        return tot

    r = minimize(_obj, np.asarray(c0, float), method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=1e-6, maxiter=n_iter))
    c = r.x
    # 曲率 → 1σ（沿 x/y 二阶差分；残差方差 = f0/(n-2)）
    h = 0.5
    f0 = _obj(c)
    n = sum(p[0].size for p in rings)
    s2 = f0 / max(n - 2, 1)
    d2x = (_obj(c + [h, 0]) - 2 * f0 + _obj(c - [h, 0])) / h ** 2
    d2y = (_obj(c + [0, h]) - 2 * f0 + _obj(c - [0, h])) / h ** 2
    sx = np.sqrt(2 * s2 / max(d2x, 1e-9)) if d2x > 0 else float("inf")
    sy = np.sqrt(2 * s2 / max(d2y, 1e-9)) if d2y > 0 else float("inf")
    return dict(cx=float(c[0]), cy=float(c[1]), sx=float(sx), sy=float(sy),
                n=len(rings), ok=bool(r.success))


def center_radial_symmetry(img, sigma=3.0, pct=85.0):
    """径向对称闭式定心（Hastings 2006/Holtzman 2010 思路 + 1031 调优）。

    原理：同心环上每点梯度方向即径向 → 过该点的梯度线必过环心；
    全体梯度线加权 LSQ 求交点，闭式解（2×2 线性方程组，零迭代零初值）。

    1031 val 域调优结论（近心 2.9px / 中离轴 18px 中位；裸版 331px！）：
    · 必须先减大尺度背景（gaussian σ25）——平背景/驼峰的常梯度把交点
      系统性拉偏；
    · 平滑 σ3 + 梯度 85 分位门 + 高于背景掩膜——宽环梯度小、噪声梯度
      大，裸 |g|² 权重被噪声接管；
    · 权重 99 分位截顶防热像素绑架；2.5σ 残差二遍鲁棒。
    · 已知系统偏差：交点被拉向像素质心 → **偏轴幅度收缩**（GT 在 2×
      画布外时只走到 1/3~1/2 路程，方向正确）→ 调用方应配合射线外推
      多种子使用（见 offset_duel5.py 臂 E）。

    返回 (cx, cy) 像素索引坐标；奇异/退化回退几何中心。
    """
    from scipy.ndimage import gaussian_filter

    img = np.asarray(img, np.float64)
    H, W = img.shape
    sm = gaussian_filter(img - gaussian_filter(img, 25.0), sigma)
    gx = ((sm[1:-1, 2:] - sm[1:-1, :-2]) * 0.5).ravel()
    gy = ((sm[2:, 1:-1] - sm[:-2, 1:-1]) * 0.5).ravel()
    yy, xx = np.mgrid[1:H - 1, 1:W - 1]
    xx = xx.astype(np.float64).ravel()
    yy = yy.astype(np.float64).ravel()
    g = np.hypot(gx, gy)
    pos = sm[1:-1, 1:-1].ravel()
    m = (g >= np.percentile(g, pct)) & (pos > 0)
    if m.sum() < 50:
        m = g >= np.percentile(g, 50)
    a, b, c = gy[m], -gx[m], gy[m] * xx[m] - gx[m] * yy[m]
    w = np.minimum(g[m], np.percentile(g[m], 99.0)) ** 2

    def _solve(w):
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

    o = _solve(w)
    if o is None:
        return (W - 1) / 2.0, (H - 1) / 2.0
    r = a * o[0] + b * o[1] - c
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    keep = np.abs(r - np.median(r)) < 2.5 * s
    if keep.sum() > 50:
        o2 = _solve(w * keep)
        if o2 is not None:
            o = o2
    return float(o[0]), float(o[1])


def gradient_vote_center(canvas, step=2.0, edge_frac=0.10, sigma=1.0,
                         acc_sigma=1.5):
    """梯度投票定心（部分弧可用），2026-11-01 独立复核 zoo 失败项时新增。

    环边缘的亮度梯度沿半径方向 → 沿 ±梯度方向按距离 r 投票，累加 (x,y)
    直方图，峰即环心。与 center_radial_symmetry（闭式最小二乘）相比：
    闭式解被像素质心拉向画面中心（偏轴收缩），投票取众数，束在画布
    一角、只剩部分弧时仍能收敛。Pilatus1M 实证：GT (60.0,88.0) canvas，
    投票给出 (59,87)（1.4px），而 46px 网格锐度定心给 (23,69)（40px 偏）。
    成本：一次 O(边缘像素×半径档)，492² 画布约 0.2s，可接受。

    返回 (cx, cy) 像素索引坐标；证据不足时回退几何中心。
    """
    from scipy.ndimage import gaussian_filter, sobel

    img = gaussian_filter(np.clip(np.asarray(canvas, np.float64), 0, None),
                          sigma)
    H, W = img.shape
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    mag = np.hypot(gx, gy)
    thr = np.quantile(mag, 1 - edge_frac)
    ys, xs = np.nonzero(mag > max(thr, 1e-9))
    if ys.size < 100:
        return (W - 1) / 2.0, (H - 1) / 2.0
    gxs, gys = gx[ys, xs], gy[ys, xs]
    n = np.hypot(gxs, gys) + 1e-12
    ux, uy = gxs / n, gys / n
    acc = np.zeros(H * W, np.float32)
    rmax = max(H, W) * 0.5
    for r in np.arange(6.0, rmax, step):
        for sgn in (-1.0, 1.0):
            cx = np.rint(xs + sgn * r * ux).astype(np.int32)
            cy = np.rint(ys + sgn * r * uy).astype(np.int32)
            ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            acc += np.bincount(cy[ok] * W + cx[ok],
                               minlength=H * W).astype(np.float32)
    acc = gaussian_filter(acc.reshape(H, W), acc_sigma)
    py, px = np.unravel_index(int(np.argmax(acc)), acc.shape)
    return float(px), float(py)


def refine_center_circles(canvas, c0, n_iter=4, max_rings=6):
    """迭代圆拟合定心（2026-11-01 晚，mar3450 攻坚新增）。

    梯度投票只能把中心压到 ~5-10px；对 mar3450（LaB6 0.374Å，内环在
    s=3 画布上只隔 12-20px）这个量级仍会把解算带进 Al2O3 假解。做法：
    以当前中心提径向谱→取最强几条环→每条环在 ±2px 带内取高亮像素做
    代数圆拟合（2.5σ 剪枝两遍）→按峰高加权聚合。4 轮内从 10px 收敛到
    <1px（实测 860.6/854.6 vs GT 861.3/854.5，s=2）。
    """
    cx, cy = float(c0[0]), float(c0[1])
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for _ in range(n_iter):
        peaks, prof, bp, good = ring_radii(canvas, cx, cy)
        if len(peaks) < 3:
            break
        r_all = np.hypot(yy - cy, xx - cx)
        cand = [(r, h) for r, h in sorted(peaks, key=lambda p: -p[1])
                if r > 40][:max_rings]
        fits = []
        for rp, h in cand:
            band = np.abs(r_all - rp) < 2.0
            iy, ix = np.nonzero(band)
            if iy.size < 150:
                continue
            v = canvas[iy, ix]
            base = float(np.percentile(v, 30))
            if h - base <= 0:
                continue
            keep = v > base + 0.4 * (h - base)
            px, py = ix[keep].astype(float), iy[keep].astype(float)
            if px.size < 100:
                continue
            kx = ky = kr = None
            sg = None
            for _ in range(2):
                A = np.stack([px, py, np.ones_like(px)], 1)
                b = px * px + py * py
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    break
                a0, b0, c0_ = sol
                kx, ky = a0 / 2.0, b0 / 2.0
                r2 = c0_ + (a0 * a0 + b0 * b0) / 4.0
                if r2 <= 0:
                    break
                kr = float(np.sqrt(r2))
                res = np.abs(np.hypot(px - kx, py - ky) - kr)
                sg = float(res.std())
                m = res < 2.5 * max(sg, 0.6)
                if m.sum() < 100 or m.all():
                    break
                px, py = px[m], py[m]
            if kx is None:
                continue
            fits.append((kx, ky, float(h), sg if sg else 9.0))
        if not fits:
            break
        sig = np.array([f[3] for f in fits])
        goodf = sig < max(1.2, np.median(sig) + 0.8)
        fits = [f for f, g in zip(fits, goodf) if g]
        if not fits:
            break
        w = np.array([f[2] for f in fits], float)
        nx = float(np.sum([f[0] * wi for f, wi in zip(fits, w)]) / w.sum())
        ny = float(np.sum([f[1] * wi for f, wi in zip(fits, w)]) / w.sum())
        if not np.isfinite(nx) or not np.isfinite(ny):
            break
        cx, cy = 0.7 * nx + 0.3 * cx, 0.7 * ny + 0.3 * cy
        last_n = len(fits)
        last_sig = float(np.median(sig[goodf]))
    return cx, cy, locals().get('last_n', 0), locals().get('last_sig', 9.0)


# ------------------------------------------------- 找环灵敏度（2026-11-06）
# 0=标准（默认，行为与旧版逐位一致）；1=高；2=极高。
# 高档降低 prominence 门并要求峰高 > 背景 + k·MAD，弱环/低对比数据能多捡几条；
# 默认档不改变任何阈值，保证回归可复现。
_RING_SENS = 0
_SENS_PROM = (1.0, 0.40, 0.15)
_SENS_K = (0.0, 0.0, 0.0)   # 不额外加噪声门：灵敏档只降显著度门


def set_ring_sensitivity(level):
    global _RING_SENS
    _RING_SENS = int(max(0, min(2, level)))


# ---------------------------------------------------------------- mask 支持
# 2026-11-06：用户可给 bool 掩膜（True=无效：束流挡盘/模块缝/坏点/遮挡）。
# 池化时按块降采样（块内 >=50% 无效 -> 画布像素无效）；径向谱与 2D 精化按
# 形状查表取用，避免把 mask 参数一路透传改遍所有调用点。
_MASK_REG = {}


def pool_mask(mask, s):
    """原生 mask(True=无效) -> 画布 mask；与 to_canvas 的裁剪池化同形状。"""
    m = np.asarray(mask, bool)
    H, W = m.shape
    s = max(int(s), 1)
    H2, W2 = H // s, W // s
    if H2 == 0 or W2 == 0:
        return np.zeros((max(H2, 1), max(W2, 1)), bool)
    blk = m[:H2 * s, :W2 * s].reshape(H2, s, W2, s)
    return blk.mean(axis=(1, 3)) >= 0.5


def register_mask(mask, s):
    if mask is None:
        return None
    mc = pool_mask(mask, s)
    _MASK_REG[tuple(mc.shape)] = mc
    return tuple(mc.shape)


def clear_masks():
    _MASK_REG.clear()


# ---------------- 模块缝自动 mask(B7,2026-11-07) ----------------
_AUTO_GAP = True     # 探针/消融可关;zoo 判决:Pilatus 全中零误报,
                     # Eiger(已校正)零动作,Frelon 织构暗带 P95 判据全排除


def auto_gap_mask(img, rel=0.05, min_w=2, max_w=80, p95_rel=0.20):
    """行列恒暗带检测 → 模块缝/边界死区 mask。

    判据(探针 _b7_gapmask_probe.py,2026-11-07 判决):
      暗带:行/列中位 < rel×全局中位(Pilatus 缝值=-1 → 必中);
      真缝:带内 P95 < p95_rel×全局中位(无信号穿带——织构暗带必有环穿过)
            或带内死0(≤0)比例>98%。
    返回 bool mask(True=无效)或 None(无真缝,零开销不注册)。
    """
    import numpy as _np
    img = _np.asarray(img, dtype=_np.float64)
    if img.ndim != 2 or min(img.shape) < 64:
        return None
    gm = float(_np.median(img))
    if gm <= 0:
        return None
    H, W = img.shape
    out = _np.zeros((H, W), bool)
    n = 0
    for axis, med in ((0, _np.median(img, axis=1)),
                      (1, _np.median(img, axis=0))):
        dark = med < rel * gm
        i = 0
        while i < len(dark):
            if dark[i]:
                j = i
                while j + 1 < len(dark) and dark[j + 1]:
                    j += 1
                w = j - i + 1
                if min_w <= w <= max_w:
                    sl = ((slice(i, j + 1), slice(None)) if axis == 0
                          else (slice(None), slice(i, j + 1)))
                    b = img[sl]
                    true_gap = (float(_np.percentile(b, 95)) < p95_rel * gm
                                or float((b <= 0).mean()) > 0.98)
                    if true_gap:
                        out[sl] = True
                        n += 1
                i = j + 1
            else:
                i += 1
    return out if n else None


def _mask_for(shape):
    return _MASK_REG.get(tuple(shape))


def ring_radii(canvas, cx, cy, nb=NB, min_r=MIN_R, prom=PROM):
    """均值径向谱 + find_peaks + 抛物线内插。返回 [(r_px, 峰高), ...] 升序。
    · 中位数方案已证伪：环像素也是少数，median 把环一起杀。
    · cnt 掩膜已证伪：ceria 真环靠角落 bin 喂票，会被 <15% 中位掩膜杀光
      （2026-10-31 回归教训）。
    · 反背景判据：角落涟漪峰高 < 背景中位（负向凹陷抖动），真环必高于
      背景 → 只收 prof[p] > bg 的峰。
    """
    H, W = canvas.shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    r = np.hypot(yy - cy, xx - cx)
    rmax = float(r.max())
    idx = (r / rmax * (nb - 1)).astype(np.int32).ravel()
    _m = _mask_for(canvas.shape)
    if _m is not None:
        valid = ~_m.ravel()
        v = np.where(valid, canvas.astype(np.float64).ravel(), 0.0)
        cnt = np.bincount(idx, weights=valid.astype(np.float64), minlength=nb)
    else:
        v = canvas.astype(np.float64).ravel()
        cnt = np.bincount(idx, minlength=nb)
    prof = np.bincount(idx, weights=v, minlength=nb) / np.maximum(cnt, 1)
    good = cnt > 0.15 * np.median(cnt[cnt > 0])
    bg = float(np.median(prof[cnt > 0]))
    bin_px = rmax / nb
    # 2026-11-01 直射束修正（独立复核 zoo 失败项时发现）：直射束/束挡边缘
    # 压在 r<min_r 时，prof.max() 由它定义，prom*max 会把所有衍射环一起
    # 压死。Pilatus1M 实证：max 在 r=6.5px（束斑，7.8e4），第 2/3 环
    # （r=108/164px，峰高 2.6e3/2.3e3）低于 0.05*max 全被丢弃，环数从 3
    # 掉到 1——"只剩 1 个完整峰"的真因就是这个阈值。直射束不是衍射峰，
    # 不参与显著度基准：改用 min_r 之外的动态范围（其余情况与旧行为等价，
    # 有束挡的生产数据 max 本来就在环上）。
    _rr = (np.arange(nb) + 0.5) * bin_px
    _outer = _rr >= min_r
    p_ref = float(prof[_outer].max()) if _outer.any() else float(prof.max())
    _lv = _RING_SENS
    _pdet = None
    if _lv > 0:
        # 去背景检测：减掉 ~51 bin 滑动中位数，陡背景上的弱环才显出来
        from scipy.ndimage import median_filter as _mf
        _pdet = prof - _mf(prof, size=51, mode="nearest")
        _ref = float(_pdet[_outer].max()) if _outer.any() else float(_pdet.max())
        pk, _ = find_peaks(_pdet, prominence=max(prom * _ref * _SENS_PROM[_lv], 1e-12),
                           distance=12)
    else:
        pk, _ = find_peaks(prof, prominence=prom * p_ref, distance=12)
    out = []
    for p in pk:
        if p <= 2 or p >= nb - 3:
            continue
        if prof[p] <= bg:            # 角落涟漪/负向抖动，非衍射峰
            continue
        if _pdet is not None and _pdet[p] <= 0:
            continue
        y0, y1, y2 = prof[p - 1], prof[p], prof[p + 1]
        d = 0.5 * (y0 - y2) / max(y0 - 2 * y1 + y2, 1e-12)
        rp = (p + 0.5 + np.clip(d, -1, 1)) * bin_px
        if _SPLIT_G:                 # B11c:分裂高斯精化(偏斜 11× 压缩)
            rg = _split_gauss_center(prof, p, bg, c0=rp / bin_px)
            if rg is not None:
                rp = (rg + 0.5) * bin_px
        if rp < min_r:
            continue
        out.append((rp, prof[p]))
    out.sort()
    # 子峰合并：噪声/倾角把一条环裂成多个近邻峰（2026-10-31 alumina 0032：
    # 首环裂 6 个子峰跨 ±9px，配对 rms 虚高 0.02px 门槛误杀真解）。
    # 阈 4.5px：子峰缝 2~4px；真双线最小间距 7.8px（Al2O3 194/202@D550）
    # ——7px 阈曾把真双线并成质心，反而毒化 rms（同日回归教训）。
    merged = []
    for rp, h in out:
        if merged and rp - merged[-1][0] < 4.5:
            rp0, h0 = merged[-1]
            merged[-1] = ((rp0 * h0 + rp * h) / (h0 + h), h0 + h)
        else:
            merged.append((rp, h))
    return merged, prof, bin_px, good


_SPLIT_G = True    # B11c/d+B12①(2026-11-07):分裂高斯定心默认开。
# 合成域抗偏斜 11×(0.3px 偏斜残余 0.179→0.026px);ceria×7+P2M 真实
# A/B 中性(ΔD≤0.03mm,无偏斜可消时无损);alumina 织构 9 帧 A/B
# (B12①):|Δ|均值 6.51→1.94mm——8/9 帧中性、0036 帧抛物线错解
# −43.5mm 被 splitG 兜住 −3.5mm。织构域稳健性 >> 对称域零成本。


def _split_gauss_center(prof, p, bg, win=6, c0=None):
    """峰 p 的分裂高斯精化(antisym 偏斜一阶免疫,B11c 0.3px→0.026px)。
    拟合 (c,a,sL,sR,b);失败/越窗/移位>1.5bin 返回 None(回退抛物线)。
    c0:抛物线中心初值(bin 索引,2026-11-08 zoo 提速:准初值少迭代)。
    2026-11-08 zoo 卡死修复②:解析 jac(省 6× 数值差分)+峰高门
    (近噪声峰直回抛物线,弱谱假峰海量时省调用)。"""
    try:
        from scipy.optimize import least_squares
        lo, hi = max(0, p - win), min(len(prof), p + win + 1)
        x = np.arange(lo, hi, dtype=float)
        y = prof[lo:hi]
        if len(x) < 7:
            return None
        if prof[p] < 2.0 * abs(bg) + 1e-12:   # 峰高门:纯噪声峰不值 5 参
            return None
        c_init = float(p) if c0 is None else float(np.clip(c0, p - 1, p + 1))
        p0 = [c_init, float(prof[p]), 2.0, 2.0, float(bg)]

        def resid(q):
            c, a, sL, sR, b = q
            s = np.where(x < c, np.abs(sL), np.abs(sR)) + 1e-6
            return a * np.exp(-0.5 * ((x - c) / s) ** 2) + b - y

        def jac(q):
            c, a, sL, sR, b = q
            sl, sr = abs(sL) + 1e-6, abs(sR) + 1e-6
            s = np.where(x < c, sl, sr)
            E = a * np.exp(-0.5 * ((x - c) / s) ** 2)
            u = (x - c) / s
            J = np.empty((len(x), 5))
            J[:, 0] = E * u / s            # ∂/∂c(忽略分支切换项,方向足够)
            J[:, 1] = E / a                # ∂/∂a
            J[:, 2] = np.where(x < c, E * u * u / sl, 0.0)   # ∂/∂sL
            J[:, 3] = np.where(x < c, 0.0, E * u * u / sr)   # ∂/∂sR
            J[:, 4] = 1.0                  # ∂/∂b
            return J

        fit = least_squares(resid, p0, jac=jac, loss="soft_l1",
                            f_scale=2.0,
                            # 2026-11-08 zoo 卡死修复:默认容差下 13 点
                            # 拟合磨 571 nfev/38ms,×λ 网格×cascade×
                            # 每解数百峰 = zoo 全量 20-60× 慢化
                            # (Pilatus1M/2M >15 分钟无输出,py-spy 实锤
                            # 本函数)。峰位需求 0.026px 级(B11c)=1e-3
                            # bin,容差 1e-4+nfev 60 已 10× 富余。
                            xtol=1e-4, ftol=1e-4, gtol=1e-4,
                            max_nfev=60,
                            bounds=([p - 1.5, 0, 0.3, 0.3, -np.inf],
                                    [p + 1.5, np.inf, 20, 20, np.inf]))
        c = float(fit.x[0])
        if not np.isfinite(c) or abs(c - p) > 1.5:
            return None
        return c - 0.5            # 返回 bin 索引(减 0.5 与 prof 网格一致)
    except Exception:
        return None


SIGMA_PK = 2.5    # 模型环宽(px@画布)——全谱相关打分用


def profile_score(prof, bin_px, good, qs, I_rel, lam, D_px,
                  rmax_use=None):
    """全谱相关：合成径向谱 vs 实测谱的 Pearson 相关。
    伪解只碰得上峰位置子集，实测谱在伪解的"预测空档"处有真环 → 相关崩。
    对织构鲁棒（位置主导），对索引错位免疫。返回 [-1,1]。"""
    nb = len(prof)
    rr = (np.arange(nb) + 0.5) * bin_px
    use = good & (rr > 30) & (rr < (rmax_use or 1.05 * rr[good][-1]))
    if use.sum() < 50:
        return 0.0
    rm = _r_model(np.asarray(qs), lam, D_px)
    model = np.zeros(nb)
    for r_i, ii in zip(rm, np.asarray(I_rel, float)):
        model += ii * np.exp(-0.5 * ((rr - r_i) / SIGMA_PK) ** 2)
    obs = prof - np.median(prof[use])
    obs = np.clip(obs, 0, None)[use]
    mod = model[use]
    if np.std(obs) < 1e-9 or np.std(mod) < 1e-9:
        return 0.0
    return float(np.corrcoef(obs, mod)[0, 1])


def _r_model(q, lam, D_px):
    tt = 2 * np.arcsin(np.clip(q * lam / (4 * np.pi), 0, 0.999))
    return D_px * np.tan(tt)


def _grid_seeds(peaks_rp, qs, lam, D_lo_px, D_hi_px, dbg=False):
    """D 网格全局扫描：每个 D 联合评估全部环的覆盖分，返回 top-3 种子。
    替代纯投票聚类——对"索引错位自洽"（q_i 整体错配 q_{i+1}，D 偏 ~10%
    仍自洽）免疫，因为网格点在真 D 处覆盖分严格更高。"""
    if not peaks_rp:
        return []
    rp = np.array([p[0] for p in peaks_rp])
    grid = np.exp(np.linspace(np.log(D_lo_px), np.log(D_hi_px), 220))
    tt = 2 * np.arcsin(np.clip(qs * lam / (4 * np.pi), 0, 0.999))
    rm = np.outer(grid, np.tan(tt))              # (nD, nq) 模型半径
    tol = np.maximum(1.5, 0.03 * rm)
    dist = np.abs(rp[None, None, :] - rm[:, :, None])   # (nD, nq, npk)
    matched = (dist < tol[:, :, None]).any(axis=2)
    m_g = matched.sum(axis=1)
    # 覆盖分：解释 +1，可见未解释 −0.5
    visible = ((rm > MIN_R) & (rm < 1.05 * rp.max())).sum(axis=1)
    score = m_g - 0.5 * (visible - m_g)
    # 局部极大（log 网格等距 → 差分）
    seeds = []
    for i in range(1, len(grid) - 1):
        if score[i] >= score[i - 1] and score[i] > score[i + 1] \
                and score[i] >= 3:
            seeds.append((score[i], grid[i]))
    seeds.sort(reverse=True)
    # 去重：同盆域只留最高分
    uniq = []
    for sc, D in seeds:
        if all(abs(np.log(D / u[1])) > np.log(1.06) for u in uniq):
            uniq.append((sc, D))
        if len(uniq) >= 3:
            break
    if dbg:
        print(f"    [dbg] 网格种子: " + ", ".join(
            f"D={D:.0f}px(s={sc:.1f})" for sc, D in uniq))
    return [float(D) for _, D in uniq]


def _center_refine(canvas, cx, cy, rm_list, I_rel, dbg=False):
    """环方位角一次谐波定心：环上像素 r(θ) ≈ r̄ − dx·cosθ − dy·sinθ。
    对最强 4 条可见环做加权回归，取中位偏移。返回 (cx', cy')。"""
    H, W = canvas.shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    r = np.hypot(yy - cy, xx - cx)
    bg = float(np.median(canvas))
    dxs, dys = [], []
    order = np.argsort(-np.asarray(I_rel, float))
    r_edge = min(cx, cy, W - 1 - cx, H - 1 - cy) * 0.95
    for k in order[:6]:
        r0 = float(rm_list[k])
        if not (40 < r0 < r_edge):
            continue
        sel = np.abs(r - r0) < 5.0
        iy, ix = np.nonzero(sel)
        w = canvas[iy, ix] - bg
        keep = w > 0
        if keep.sum() < 200:
            continue
        w = w[keep]
        th = np.arctan2(iy[keep] - cy, ix[keep] - cx)
        rr = r[iy[keep], ix[keep]]
        A = np.stack([np.cos(th), np.sin(th)], axis=1)
        Wd = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * Wd[:, None],
                                   (r0 - rr) * Wd, rcond=None)
        dxs.append(float(np.clip(coef[0], -15, 15)))
        dys.append(float(np.clip(coef[1], -15, 15)))
        if len(dxs) >= 4:
            break
    if not dxs:
        return cx, cy
    dx, dy = float(np.median(dxs)), float(np.median(dys))
    if dbg:
        print(f"    [dbg] 定心: dx={dx:+.1f} dy={dy:+.1f} "
              f"(环数{len(dxs)}, 各环 {[f'{a:+.1f},{b:+.1f}' for a, b in zip(dxs, dys)]})")
    return cx + dx, cy + dy


def solve(peaks_rp, cx0, cy0, D_px0, q_list, lam, px_m,
          dbg=False, D_ref_m=0.0, d_range=(0.05, 3.0), I_rel=None,
          prof_ctx=None, rms_floor=3.5, canvas=None, recenter=True):
    """单相求解。D_px0 仅用于投票范围锚（无 NN 时给画布对角即可，
    d_range 给宽倍率）。canvas 给定时启用环谐波定心迭代（关键：LSQ 中
    cx,cy 是零方向，峰半径全量自初始中心——中心偏会让环裂成子峰、
    rms 虚高、真解被 rms 门误杀；定心后重提取峰再解，2026-10-31）。
    返回 (cx, cy, D_px, m, rms, icorr, pscore) 或 None。"""
    lam = _lam_A(lam)
    qs = np.asarray(q_list, float)
    I_rel = (np.ones(len(qs)) if I_rel is None
             else np.asarray(I_rel, float))
    votes = []
    for rp, _ in peaks_rp:
        for q in qs:
            tt = 2 * np.arcsin(np.clip(q * lam / (4 * np.pi), 0, 0.999))
            D = rp / np.tan(tt)
            if d_range[0] * D_px0 < D < d_range[1] * D_px0:
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
    # 种子 = 投票簇 top-5（2026-11-01 zoo 跨仪器教训：top-3 会让真相 D
    # 被 3 个谐波簇挤出（mar3450 LaB6 真 D≈318px 被 755/795/855px 谐波
    # 全占）；top-5 + rms/m 门过滤谐波，真相解得以入场）
    clusters = [c for c in clusters if c[0] >= 3][:5]
    if not clusters:
        if dbg:
            print("    [dbg] 无≥3票簇")
        return None
    seeds = [float(D) for _, D in clusters]
    # 密环域补充种子（2026-11-01 W9，Eiger9M/mar3450 短λ 27峰取证）：
    # 峰数≥15 时投票簇被索引错位谐波占满（真 D 挤不出场），追加 D 网格
    # 全局扫描 top-2——网格对索引错位免疫（真 D 处覆盖分严格更高）。
    # 阈 15 保证 alumina（≤14峰）等生产域不触发。
    if len(peaks_rp) >= 999:  # bisect: disabled
        d_hi = d_range[1] * D_px0
        d_lo = d_range[0] * D_px0
        for Dg in _grid_seeds(peaks_rp, qs, lam, d_lo, d_hi, dbg=dbg)[:2]:
            if all(abs(np.log(Dg / s0_)) > np.log(1.06) for s0_ in seeds):
                seeds.append(float(Dg))

    def pair_up(cx, cy, D, frac):
        # 双射配对（2026-11-01 zoo 修复）：全局按距离贪心，一条 q 用一个
        # 峰、一个峰只配一条 q。旧版每 q 独立取最近峰 → 密线相（Quartz
        # 15 线）在少量真峰上"多对一"复用刷高 m/覆盖分（Eiger9M 伪 m=14
        # 实为 ~9 峰被重复占用），稀疏真相相必输。双射下 m ≤ 峰数，恢复
        # 公平比较。
        cand = []
        for qi, q in enumerate(qs):
            rm = _r_model(np.array([q]), lam, D)[0]
            tol = max(1.5, frac * rm)
            for pi, (rp, h) in enumerate(peaks_rp):
                if abs(rp - rm) < tol:
                    cand.append((abs(rp - rm), qi, pi, rp, h))
        cand.sort()
        used_q, used_p = set(), set()
        pairs = []
        for _d, qi, pi, rp, h in cand:
            if qi in used_q or pi in used_p:
                continue
            used_q.add(qi)
            used_p.add(pi)
            pairs.append((qs[qi], rp, h, I_rel[qi]))
        return pairs

    def refine_from(peaks_in, cx_in, cy_in, D_star):
        cx, cy, D = cx_in, cy_in, D_star
        best = None
        for frac in (0.03, 0.02, 0.01):
            pairs = pair_up(cx, cy, D, frac)
            if len(pairs) >= 3:
                qv = np.array([q for q, _, _, _ in pairs])
                rv = np.array([rp for _, rp, _, _ in pairs])
                hv = np.array([h for _, _, h, _ in pairs])
                iv = np.array([i for _, _, _, i in pairs])

                def resid(p):
                    return _r_model(qv, lam, p[2]) - rv

                s = least_squares(resid, [cx, cy, D], method="lm")
                cx, cy, D = s.x
                rms = float(np.sqrt(np.mean(s.fun ** 2)))
                icorr = _icorr(iv, hv)
                if best is None or (len(pairs), -rms) > (best[3], -best[4]):
                    best = (float(s.x[0]), float(s.x[1]), float(s.x[2]),
                            len(pairs), rms, D_star, icorr)
        return best

    def refine(D_star):
        return refine_from(peaks_rp, cx0, cy0, D_star)

    cands = []
    rmax_est = 1.05 * max(rp for rp, _ in peaks_rp) if peaks_rp else 0.0
    for D_star in seeds:
        r = refine(D_star)
        if r is None:
            if dbg:
                print(f"    [dbg] 种子 D*={D_star:.1f} 配对<3")
            continue
        # —— 环谐波定心迭代（≤2 轮）：定心 → 重提取峰 → 重解 ——
        if canvas is not None and recenter and r[3] >= 4:
            cx, cy, cur = cx0, cy0, r
            for _it in range(2):
                cx2, cy2 = _center_refine(
                    canvas, cx, cy, _r_model(qs, lam, cur[2]), I_rel,
                    dbg=dbg)
                if abs(cx2 - cx) + abs(cy2 - cy) < 0.5:
                    break
                peaks2, prof2, bp2, g2 = ring_radii(canvas, cx2, cy2)
                cur2 = refine_from(peaks2, cx2, cy2, cur[2])
                if cur2 is not None and (cur2[3], -cur2[4]) >= (cur[3], -cur[4]):
                    cx, cy, cur = cx2, cy2, cur2
                else:
                    break
            r = (cx, cy) + tuple(cur[2:])
        cx, cy, D, m, rms, Ds, icorr = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        basin_ok = (D_ref_m <= 0
                    or abs(D * px_m - D_ref_m) < BASIN_M)
        # rms 门槛：相对 1.2% + 绝对下限 3.5px（下限须够宽——
        # 按候选自身半径缩放会系统性偏袒大 D 错解，2026-10-31 alumina 教训）
        rms_lim = max(rms_floor, 0.012 * float(np.median(
            _r_model(qs, lam, D))))
        rm_c = _r_model(qs, lam, D)
        visible = int(((rm_c > MIN_R) & (rm_c < rmax_est)).sum())
        cov_score = m - 0.5 * (visible - m)
        pscore = (profile_score(prof_ctx[0], prof_ctx[1], prof_ctx[2],
                                qs, I_rel, lam, D, rmax_use=rmax_est)
                  if prof_ctx is not None else -1.0)
        if dbg:
            print(f"    [dbg] 种子 D*={Ds:.1f} 配 {m}/{visible} 环 "
                  f"rms={rms:.2f}/{rms_lim:.2f}px Icorr={icorr:+.2f} "
                  f"谱相关={pscore:+.3f}(诊断) D_fit={D:.1f} "
                  f"覆盖分={cov_score:.1f} 盆域={'✓' if basin_ok else '✗'}")
        if not basin_ok:
            continue
        if m == 3 and abs(D - Ds) / Ds > 0.03:
            continue
        if m >= 4 and rms > rms_lim:
            continue
        cands.append((cov_score, m, -rms, cx, cy, D, icorr, pscore))
    if not cands:
        return None
    cands.sort(reverse=True)
    cs, m, neg_rms, cx, cy, D, icorr, ps = cands[0]
    return (cx, cy, D, m, -neg_rms, icorr, ps)


def _icorr(iv, hv):
    """理论强度 vs 实测峰高 Pearson 相关（防伪相自洽）。"""
    if len(iv) < 4 or np.std(iv) < 1e-9 or np.std(hv) < 1e-9:
        return 0.0
    return float(np.corrcoef(iv, hv)[0, 1])


def consensus_pick(cands, d_tol=0.02, c_tol=12.0):
    """跨种子共识门（11 号文档 §六.1 开放问题的候选解法）。

    动机：两域标量门反例已钉死——干净域 m-first 选中射线伪解（m8 骗过
    真解 m5~6），离轴域 pscore 反向误判（真解 ps −0.002 vs 伪解 +0.723）。
    共识的物理依据：**真盆地会被多颗种子重复发现**（不同起点收敛到同一
    (cx,cy,D)），伪解彼此独立、各说各话。

    票数(i) = #{j: |D_i−D_j|/D_i < d_tol 且 中心距 < c_tol}（含自身）。
    择优：票数↓ → pscore↓ → m↓ → rms↑。
    cands: [(tag, sol)]，sol 为 solve 返回的 7 元组；返回 (tag, sol, votes)
    或 None（空入参）。阈值为 490px 画布经验值，未标定灵敏度——调用方
    应自行回归（cascade_probe / offset_duel5 两域判决台）。
    """
    ok = [(t, s) for t, s in cands if s is not None and s[3] >= 3]
    if not ok:
        return None
    votes = []
    for i, (ti, si) in enumerate(ok):
        v = 0
        for j, (tj, sj) in enumerate(ok):
            if j == i:
                v += 1
                continue
            if (abs(si[2] - sj[2]) / max(si[2], 1e-9) < d_tol
                    and np.hypot(si[0] - sj[0], si[1] - sj[1]) < c_tol):
                v += 1
        votes.append(v)
    k = max(range(len(ok)),
            key=lambda i: (votes[i], ok[i][1][6], ok[i][1][3],
                           -ok[i][1][4]))
    return ok[k][0], ok[k][1], votes[k]


def solve_auto(canvas, lam, px_m, phases=None, dbg=False,
               D_ref_m=0.0, cx0=None, cy0=None, d_range=(0.05, 3.0),
               rms_floor=3.5, cascade=False, auto_center=False,
               refine_2d=False):
    """多相自动识别（全托管）：画布 → 峰 + 径向谱 → 逐相 solve →
    按【全谱相关 pscore ↓ 覆盖分 ↓ rms ↑】全局择优。
    pscore：合成谱 vs 实测谱 Pearson 相关——伪相/索引错位解只碰得上
    峰位置子集，实测谱在其"预测空档"处的真环令相关崩盘（2026-10-31）。
    cascade=True 时逐相走 solve_cascade（经典优先 + 失败兜底），成本
    ≈种子数倍（默认 False，离线评估用，见 11 号文档耗时表）。
    auto_center=True（2026-11-01 新增，默认关）：束心离画面几何中心
    >25px（束打在探测器一角/一侧）时，额外用梯度投票定心跑一遍，只有
    谱相关领先 >0.25 才取代几何中心解——治"束在画布一角"的零样本失败
    （Pilatus1M：+0.885 vs +0.291 放行；Eiger4M：+0.360 vs +0.444 挡住）。
    生产路径不传，行为不变。
    refine_2d=True（2026-11-01 深夜 T3，默认关）：解出后若 ps>0.4 且
    m≥4，用 refine_geometry_2d 逐环 Kasa 圆拟合精化中心与 D（mar3450
    中心敏感坑的根治位）。精化结果替换 sol[0..2]，sol 尾部追加
    (pre_cx, pre_cy, pre_D_px, n_rings) 供基准记录前后差；旧调用只读
    sol[:7] 不受影响。生产不传，行为不变。
    返回 (phase_name, sol) 或 None，sol=(cx,cy,D_px,m,rms,icorr,pscore)。"""
    H, W = canvas.shape
    if cx0 is None:
        cx0 = (W - 1) / 2.0
    if cy0 is None:
        cy0 = (H - 1) / 2.0
    res = _solve_auto_at(canvas, lam, px_m, cx0, cy0, phases, dbg,
                         D_ref_m, d_range, rms_floor, cascade)
    if auto_center:
        # 一、梯度投票给粗中心（部分弧可用，抗像素质心拉偏）；
        # 二、迭代圆拟合把粗中心压到 ~1px（mar3450 实证 6.9px→1.1px）——
        #     细环探测器在 5px 中心误差下就会把 LaB6 解成 Al2O3 假解。
        gcx, gcy = gradient_vote_center(canvas)
        rcx, rcy, qn, qs = refine_center_circles(canvas, (gcx, gcy))
        if qn >= 3 and qs < 1.0:
            alt = _solve_auto_at(canvas, lam, px_m, rcx, rcy, phases, dbg,
                                 D_ref_m, d_range, rms_floor, cascade,
                                 trust_center=True)
        else:
            alt = _solve_auto_at(canvas, lam, px_m, gcx, gcy, phases, dbg,
                                 D_ref_m, d_range, rms_floor, cascade,
                                 trust_center=True)
        # 取代条件：谱相关更高且 rms 更小（两个独立证据都改善）。
        # Pilatus1M：geo 处 LaB6 ps+0.722/rms0.88（索引错位，7/12），
        # 精定心中心处 AgBh ps+0.885/rms0.39（3/3）→ 放行；
        # Eiger4M：geo ps+0.444/rms0.31 vs 投票 ps+0.360/rms3.05 → 挡住。
        if alt is not None and (res is None
                                or (alt[1][6] > res[1][6]
                                    and alt[1][4] < res[1][4])):
            res = alt
    if refine_2d and res is not None and res[1][6] > 0.4 and res[1][3] >= 4:
        # T3：2D 逐环 Kasa 精化（门控 ps>0.4 且 m≥4——初值错会收敛到
        # 错环族；refine_geometry_2d 自带 <3 环 / 中心跳 >30px 拒收）。
        lamA = _lam_A(lam)
        out2 = refine_geometry_2d(canvas, res[1][0], res[1][1], res[1][2],
                                  phase_qs(res[0]), lamA, dbg=dbg)
        if out2 is not None:
            pre = (res[1][0], res[1][1], res[1][2])
            sol = list(res[1][:7])
            sol[0], sol[1], sol[2] = out2[0], out2[1], out2[2]
            res = (res[0], tuple(sol) + pre + (out2[3],))
    return res


def _solve_auto_at(canvas, lam, px_m, cx0, cy0, phases, dbg, D_ref_m,
                   d_range, rms_floor, cascade, trust_center=False):
    from torchfai.phases import PHASES
    peaks, prof, bin_px, good = ring_radii(canvas, cx0, cy0)
    prof_ctx = (prof, bin_px, good)
    D_px0 = 1.0 / px_m
    lamA = _lam_A(lam)
    rmax = 1.05 * max(rp for rp, _ in peaks) if peaks else 0.0
    cands = []
    for ph in (phases or PHASE_NAMES):
        q = phase_qs(ph)
        _, I, _ = PHASES[ph]
        if dbg:
            print(f"  -- 相 {ph} ({len(q)} 环) --")
        if cascade:
            s = solve_cascade(canvas, lam, px_m, q, I,
                              center=(cx0, cy0), d_range=d_range,
                              rms_floor=rms_floor, classic_first=True,
                              dbg=dbg, trust_center=trust_center)
        else:
            s = solve(peaks, cx0, cy0, D_px0, q, lam, px_m,
                      dbg=dbg, D_ref_m=D_ref_m, I_rel=I, prof_ctx=prof_ctx,
                      d_range=d_range, rms_floor=rms_floor, canvas=canvas)
        if s is None:
            continue
        rm = _r_model(np.asarray(q), lamA, s[2])
        visible = int(((rm > MIN_R) & (rm < rmax)).sum())
        cov = s[3] - 0.5 * (visible - s[3])
        if dbg:
            print(f"    [dbg] {ph}: D={s[2]*px_m*1000:.1f}mm m={s[3]}/"
                  f"{visible} rms={s[4]:.2f} 谱相关={s[6]:+.3f} 覆盖分={cov:.1f}")
        # 覆盖分↓（m 裸数偏向线多的相）→ 谱相关↓ → 环数↓ → rms↑
        cands.append((cov, s[6], s[3], -s[4], ph, s, visible))
    if not cands:
        return None
    # 两段相仲裁（2026-11-01 W9）：①谱相关 margin 0.15 先砍——伪相索引
    # 错位解只能解释峰位子集，全谱相关必输（Pilatus2M：LaB6 伪解 cov 9.0
    # 但 ps +0.179 vs CeO2 真解 ps +0.446）；②幸存者按覆盖分比信息量。
    # 前提是层0已修：谱在锐度中心下可信（错中心垃圾谱曾令 pscore 误判，
    # 2026-10-31 注记——那时没有 center_by_sharpness）。
    ps_max = max(c[1] for c in cands)
    keep = [c for c in cands if c[1] >= ps_max - 0.15]
    keep.sort(reverse=True)
    # 子集简并破除（W9 kasa 战役）：cov 打平（≤1.0）且都过了谱相关档的
    # 两个相，偏好可见线更多的富相——Si(7线)⊂CeO2(9线) 在 Eiger9M 上
    # 0.4% 内重合，弱环一丢 Si 就以 7/7 完美纯度反超 7/9 的真相；子集相
    # 是富相的退化索引，富相解释更多谱线。Quartz 类杂相过不了①档不适用。
    if len(keep) >= 2 and keep[0][0] - keep[1][0] <= 1.0 \
            and keep[1][6] > keep[0][6]:
        keep[0], keep[1] = keep[1], keep[0]
    return keep[0][4], keep[0][5]


def refine_lambda_d(canvas, cx, cy, D_px, lam, q_list, dbg=False,
                    lam_frac=0.06, sig_px=0.25, sector=False):
    """T1 连续 λ–D 收口：固定相与中心，匹配环半径对 (λ, D_px) 加权 LSQ。

    为什么需要它：网格 + 离散评分 (ps, m, −rms) 在简并脊上分不出 λ
    （Pilatus1M 实证：ps 全程 +0.859 平坦、rms 差 0.01 噪声级 → λ̂ 沿
    脊随机游走到 1.153 而非 1.033）。但环半径 r_i = D·tan(2θ(q_i,λ))/px
    的宽角非线性对每条环不同——r_i/r_j 与 D 无关、只由 λ 决定。σ(λ) 由
    JᵀJ 反演给出诚实误差棒，σ(λ)/λ>2% 视为不可辨（环太少/角度太窄），
    调用方应回退网格 λ̂ 并标"简并"。1D 路径战绩：Eiger4M λ̂=0.9214
    σ<2% 放行；halfccd/Frelon2k 亦过门控。

    sector=True（保留给 T2/T3 倾角感知版）：36 扇区控制点半径。深夜
    实测对 mar3450 无效——倾角 r(χ) 系统调制（rot2=43.7mrad）令环
    中位半径有环间差异偏差（rms 2.3px、σ=44% 仍被拒）；离轴部分弧
    （Pilatus1M）中位偏置同理。λ 拆分到 0.5% 需要倾角联合几何，1D/
    扇区圆模型是它的域界。默认 False。

    返回 (λ̂, D̂_px, n_use, rms_px, σ_λ/λ) 或 None。
    """
    qs = np.asarray(q_list, float)
    rm0 = _r_model(qs, lam, D_px)
    pairs = []                       # (q, r, sig)
    if sector:
        try:
            from poni_out import sector_peaks
        except Exception:
            return None
        rmax = float(np.hypot(*canvas.shape))
        vis_idx = [k for k, r0 in enumerate(rm0)
                   if MIN_R < r0 < 0.98 * rmax]
        if len(vis_idx) < 3:
            return None
        pts = sector_peaks(canvas, cx, cy, rm0[vis_idx], nsec=36,
                           win=6.0, mass_rel=0.10, bump=1.5)
        by_ring = {}
        for k, _chi, rp, _ms in pts:
            by_ring.setdefault(int(k), []).append(rp)
        for j, k in enumerate(vis_idx):
            rps = np.asarray(by_ring.get(j, []), float)
            if len(rps) < 8:         # <8 扇区的弧不做 λ 证据
                continue
            r_med = float(np.median(rps))
            mad = float(np.median(np.abs(rps - r_med))) * 1.4826
            sig = max(mad / np.sqrt(len(rps)), 0.15)
            pairs.append((qs[k], r_med, sig))
    else:
        peaks, _, _, _ = ring_radii(canvas, cx, cy)
        if len(peaks) < 3:
            return None
        rmax = 1.05 * max(rp for rp, _ in peaks)
        for qi, r0 in zip(qs, rm0):
            if r0 < MIN_R or r0 > rmax:
                continue
            jj = int(np.argmin([abs(p[0] - r0) for p in peaks]))
            if abs(peaks[jj][0] - r0) > max(1.5, 0.025 * r0):
                continue
            pairs.append((qi, peaks[jj][0], sig_px
                          / np.sqrt(max(peaks[jj][1] / max(
                              [p[1] for p in peaks]), 1e-9))))
    if len(pairs) < 3:
        return None
    q = np.array([p[0] for p in pairs])
    r = np.array([p[1] for p in pairs])
    w = 1.0 / np.array([p[2] for p in pairs])

    def resid(p):
        tt = 2 * np.arcsin(np.clip(q * p[0] / (4 * np.pi), 0, 0.9999))
        return w * (p[1] * np.tan(tt) - r)

    out = least_squares(resid, x0=[lam, D_px],
                        bounds=([lam * (1 - lam_frac), 0.3 * D_px],
                                [lam * (1 + lam_frac), 3.0 * D_px]),
                        method="trf", xtol=1e-12, ftol=1e-12)
    lam_h, D_h = float(out.x[0]), float(out.x[1])
    if not np.isfinite(lam_h) or not np.isfinite(D_h):
        return None
    # 误差棒：cov = (JᵀJ)⁻¹ · s²（自由度 n−2；n=3 时 s² 由唯一残差定）
    n = len(q)
    try:
        J = np.asarray(out.jac)
        JTJ = J.T @ J
        s2 = max(2.0 * out.cost / max(n - 2, 1), 1e-12)
        cov = np.linalg.inv(JTJ) * s2
        sig_rel = float(np.sqrt(max(cov[0, 0], 0.0)) / lam_h)
    except Exception:
        sig_rel = None
    rms = float(np.sqrt(np.mean((resid([lam_h, D_h]) / w) ** 2)))
    if dbg:
        print(f"    [λ-D LSQ{'·扇区' if sector else ''}] n={n} "
              f"λ={lam:.4f}→{lam_h:.4f} D={D_px:.1f}→{D_h:.1f}px "
              f"rms={rms:.3f}px "
              f"σ(λ)/λ={sig_rel if sig_rel is None else round(sig_rel, 4)}")
    return lam_h, D_h, n, rms, sig_rel


def refine_geometry_2d(canvas, cx, cy, D_px, q_list, lam, n_iter=3,
                       half=3.0, trim=2.5, dbg=False):
    """T3（2026-11-01 深夜）：相感知 2D 逐环 Kasa 圆拟合精化中心与 D。

    与 kasa_center 的分工：那边相无关、用径向谱最强 6 环做**定心**；
    这里用 1D 解的 (相, D, λ) 预测每条环半径 r_i=D·tan(2θ(q_i,λ))，
    逐环带内亮像元（35/98 百分位门 0.35）Kasa 圆拟合（2.5σ 剪枝×2），
    环像元数加权聚合中心（0.5 阻尼）+ 环半径对 D 的加权 LSQ
    （w∝1/r²，相对精度同权），迭代 n_iter 轮——**既精中心也精 D**，
    且不受 1D 径向谱对 1-2px 中心扰动敏感的坑（mar3450 实证）。

    原型战绩（probes/_mar_2d.py，mar3450 原生 3450²、LaB6 12 环）：
    GT 初始 0.18px、5px 扰动 1.1px、逐环 D_i 一致 ±0.6mm。
    门控由调用方负责（ps>0.4 且 m≥4 才启用——初值错会收敛到错环）。
    返回 (cx2, cy2, D_px2, n_rings) 或 None（可拟合环 <3 / 中心跳
    >30 画布px，视为收敛到错环族）。
    """
    H, W = canvas.shape
    qs = np.asarray(q_list, float)
    c0 = (float(cx), float(cy))
    c, D = c0, float(D_px)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    fits = []
    for _ in range(n_iter):
        radii = _r_model(qs, lam, D)
        fits = []
        r_edge = min(c[0], c[1], W - 1 - c[0], H - 1 - c[1])
        for qi, r0 in enumerate(radii):
            # 边缘门（冒烟实证：部分弧把圆心拉偏，Eiger4M 4.99px 的主因）：
            # 只收完整环，0.9 留倾斜/畸变余量（kasa_center 用 0.95）。
            if r0 < MIN_R or r0 > 0.9 * r_edge:
                continue
            band = np.abs(np.hypot(xx - c[0], yy - c[1]) - r0) < half
            _m2 = _mask_for(band.shape)
            if _m2 is not None:
                band = band & ~_m2
            iy, ix = np.nonzero(band)
            if iy.size < 200:
                continue
            v = canvas[iy, ix]
            base = float(np.percentile(v, 35))
            peak = float(np.percentile(v, 98))
            if peak <= base:
                continue
            keep = v > base + 0.35 * (peak - base)
            px_, py_ = (ix[keep].astype(float), iy[keep].astype(float))
            # 强度加权（冒烟实证：织构/环上亮度不对称把等权 Kasa 圆心
            # 拉向亮侧；w∝v−base，行乘 √w 进法方程）。
            wt = np.clip(v[keep] - base, 1e-9, None)
            if px_.size < 150:
                continue
            cxk = cyk = rk = sig = None
            for _ in range(2):
                sw = np.sqrt(wt)
                A = np.stack([px_, py_, np.ones_like(px_)], axis=1) \
                    * sw[:, None]
                b = (px_ ** 2 + py_ ** 2) * sw
                try:
                    sol_, *_ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    break
                a0, b0, cc = sol_
                cxk, cyk = a0 / 2.0, b0 / 2.0
                rk2 = cc + (a0 * a0 + b0 * b0) / 4.0
                if rk2 <= 0:
                    break
                rk = float(np.sqrt(rk2))
                res = np.abs(np.hypot(px_ - cxk, py_ - cyk) - rk)
                sig = float(res.std())
                m = res < trim * max(sig, 0.5)
                if m.sum() < 150 or m.all():
                    break
                px_, py_, wt = px_[m], py_[m], wt[m]
            if cxk is None or rk is None:
                continue
            res = np.abs(np.hypot(px_ - cxk, py_ - cyk) - rk)
            fits.append((cxk, cyk, rk, float(px_.size),
                         float(res.std()), qi))
        if len(fits) < 3:
            return None
        w = np.array([f[3] for f in fits], float)
        nx = float(np.sum([f[0] * wi for f, wi in zip(fits, w)]) / w.sum())
        ny = float(np.sum([f[1] * wi for f, wi in zip(fits, w)]) / w.sum())
        rr = np.array([f[2] for f in fits], float)
        qq = qs[[f[5] for f in fits]]
        tt = 2.0 * np.arcsin(np.clip(qq * lam / (4.0 * np.pi), 0, 0.999))
        k = np.tan(tt)
        ww = 1.0 / np.maximum(rr, 1.0) ** 2
        D = float(np.sum(ww * rr * k) / np.sum(ww * k * k))
        c = (0.5 * nx + 0.5 * c[0], 0.5 * ny + 0.5 * c[1])
        if dbg:
            print(f"    [2D] rings={len(fits)} c=({c[0]:.2f},{c[1]:.2f}) "
                  f"D={D * 1.0:.1f}px dc={np.hypot(c[0]-c0[0], c[1]-c0[1]):.2f}"
                  f" sig_med={np.median([f[4] for f in fits]):.2f}")
    if np.hypot(c[0] - c0[0], c[1] - c0[1]) > 30.0:
        return None
    return float(c[0]), float(c[1]), float(D), len(fits)


def solve_auto_lambda(canvas, px_m, lam_lo=0.12, lam_hi=2.50, n_s=24,
                      dbg=False, auto_center=False, refine=True,
                      refine_stages=((0.08, 7), (0.015, 7)), full=False):
    """联合 λ–D 精化（W9 v1 + T1 两段收口，2026-11-01）。

    物理：小角域 λ–D 严格简并（D·λ=常数族，几何等价，验证自然通过）；
    宽角环系 tan(2asin) 非线性弱破简并——真 (λ,D) 的全谱相关/rms 占优。
    跨 λ 仲裁键 (pscore, m, −rms)：pscore 是同一张谱的全谱相关，跨 λ
    可比；简并族内任取一员几何等价。

    T1 改动（16 号文档）：粗网格 geomspace(0.12,2.5,24) 步长 ~15%，
    λ̂ 落不到真值上（Pilatus1M 真尺度 s=1.0326 正好夹在两格之间）。
    改三段：①粗网格照旧，记 top-3 候选 (λ, ps, m, rms)；②最优 λ
    ±frac₁ 局部 n 点（去重）；③再 ±frac₂ 收口。默认两段
    ((0.08,7),(0.015,7))：末段步长 0.5% → λ̂ 误差 ≤0.25%（T1 验收
    <0.5%）。有 λ 分支的 λ̂ 报告模式用窄窗 [0.90,1.10]×λ₀、n_s=5、
    stages=((0.04,7),(0.008,7))（末段步长 0.27%）。
    refine=False 退回旧行为（只跑粗网格）。

    返回 (phase, sol, λ_Å) 或 None（兼容旧调用）；full=True 时返回
    dict(phase, sol, lam, top3=[(ph, sol, lam)×3], n_eval)。
    """
    def _try(lam):
        try:
            return solve_auto(canvas, float(lam), px_m, cascade=True,
                              auto_center=auto_center)
        except Exception:
            return None

    cands = []
    for lam in np.geomspace(lam_lo, lam_hi, n_s):
        r = _try(lam)
        if r is None:
            continue
        ph, sol = r
        key = (sol[6], sol[3], -sol[4])
        cands.append((key, ph, sol, float(lam)))
        if dbg:
            print(f"  [λ={lam:.4f}] {ph}: m={sol[3]} rms={sol[4]:.2f} "
                  f"ps={sol[6]:+.3f} D={sol[2]*px_m*1000:.1f}mm")
    if not cands:
        return None
    cands.sort(key=lambda c: c[0], reverse=True)
    best = cands[0]
    top3 = [(ph, sol, lam) for _k, ph, sol, lam in cands[:3]]
    seen = {best[3]}
    for frac, npts in (refine_stages if refine else ()):
        lam0 = best[3]
        for lam in np.linspace(lam0 * (1.0 - frac), lam0 * (1.0 + frac),
                               npts):
            lam = float(lam)
            if lam <= 0.0 or any(abs(lam - x) < 1e-9 * lam for x in seen):
                continue
            seen.add(lam)
            r = _try(lam)
            if r is None:
                continue
            ph, sol = r
            key = (sol[6], sol[3], -sol[4])
            if dbg:
                print(f"  [精化 λ={lam:.4f}] {ph}: m={sol[3]} "
                      f"rms={sol[4]:.2f} ps={sol[6]:+.3f} "
                      f"D={sol[2]*px_m*1000:.1f}mm"
                      + ("  ← best" if key > best[0] else ""))
            if key > best[0]:
                best = (key, ph, sol, lam)
    # ---- T1 连续收口：离散评分分不出简并脊上的 λ，LSQ 直接吃半径比 ----
    lam_grid, sol_grid, ph_best = best[3], best[2], best[1]
    lam_out, sol_out, src, sig = lam_grid, sol_grid, "grid", None
    lr = None
    try:
        # lam_frac=0.20：网格 λ̂ 在简并脊上的落点本来任意（Pilatus1M 粗
        # 网格可停在 1.15 而真值 1.03），LSQ 要能沿脊走全程。
        lr = refine_lambda_d(canvas, sol_grid[0], sol_grid[1], sol_grid[2],
                             lam_grid, phase_qs(ph_best), dbg=dbg,
                             lam_frac=0.20)
    except Exception:
        lr = None
    if lr is not None:
        lam_h, D_h, n_use, rms_h, sig_l = lr
        sig = sig_l
        # σ(λ)/λ≥2% = 环太少/角太窄，λ–D 数学简并（Pilatus1M 实证
        # σ=4.8：3 个小角谐波环无非线性信号）——不偷换 λ̂，如实报 grid
        # 落点 + σ，由调用方标"简并"。
        if sig_l is None or sig_l < 0.02:
            sol_out = tuple(list(sol_grid[:2]) + [D_h] + list(sol_grid[3:]))
            lam_out, src = lam_h, "lsq"
    if full:
        return dict(phase=ph_best, sol=sol_out, lam=lam_out,
                    lam_grid=lam_grid, lam_src=src, lam_sig=sig,
                    top3=[(ph, s, lam) for ph, s, lam in top3],
                    n_eval=len(seen))
    return ph_best, sol_out, lam_out


# ============ 18-1104 需求：画布策略与 fallback 统一（R1/R2/R3） ============
# 背景：三处各写一套画布（生产 to_canvas 520 / zoo_run.canvas_of 723 /
# _fallback_joint ceil(maxdim/2)）+ fallback 门控依赖 GT（"相错 AND ps<0.55"）
# → 同一算法同一图给出不同答案（ablation8v2：P2M −12.39 / E4M +244.7）。
# 统一后：zoo_run / phase_fit / gui_solver / ablation8 只调 solve_image()。

CANVAS_MAX_SIDE = 520     # R1 默认档：生产回归口径（ceria 0.1/0.1mm、
                          # alumina 116.79/MAD 0.39、zoo 四胜全建立在此）
FB_RMS_GATE = 3.5         # R2(b)：默认档 rms 超此 AND m<FB_M_GATE → 救援扫描
                          # （3.5=solve 自身 rms_floor；cascade 兜底路径可放
                          # 行 rms 超限解，"高残差+环少"=已知弱）
FB_M_GATE = 3             # R2(b)：m 严格小于此（=solve 的 m≥3 最低可用门）
FB_S_RISK = 5             # R2(c) 预门：默认档 s≥5 时池化把内环间距压进
                          # ring_radii 合并阈危险区（mar3450 s=7 → LaB6 内环
                          # 5-7px 落 4.5px 阈 → Quartz 假解；s≤4 环间距≥8px 安全）
FB_FORK = 0.10            # R2(c)：两档 D 相对分叉阈。校准（_r2_calib2）：
                          # E4M/E9M 大档为"同解重解"——Δps +0.154/+0.245 但
                          # D 分叉仅 0.2%/0.07%（ps 虚涨是分辨率换算，不是
                          # 画布档冲突）；mar3450 环合并假解 D 分叉 108%。
                          # 取 10%：与两类相隔两个量级。
FB_PS_MARGIN = 0.05       # 需求原案 (c) 的 ps 提升门——校准推翻（见
                          # unify_canvas_fallback.md），仅留档不参与判决
FB_AREA_MAX = 3_500_000   # 升级档面积上限（Fairchild 2048²≈4.2Mpx² 不扫）
FB_LAM_GRID = (0.95, 1.05, 7)   # 救援网格 geomspace×λ₀（T4 实证步长
                                # 1.68%：mar 盆地宽 1.4% 必有格点落盆）


def canvas_of(img, tier="default"):
    """R1 画布策略唯一入口（2026-11-05，18-1104 需求）。

    tier="default"：max_side=520 —— 生产回归值，ceria 0.1/0.1、alumina
        116.79/MAD 0.39、zoo 四胜全部建立在此档；禁止各处自改。
    tier="large"：max_side=⌈maxdim/2⌉ —— 升级档，仅 solve_image 在 R2
        门控触发时使用（mar3450 s=7 默认档 Quartz 假解 → s=2 分得开，
        ΔD@λ̂ +239→+5.87mm）。旧 zoo_run.canvas_of(723) 作废。
    """
    if tier == "default":
        return to_canvas(img, max_side=CANVAS_MAX_SIDE)
    if tier == "large":
        return to_canvas(img, max_side=int(np.ceil(max(img.shape) / 2.0)))
    raise ValueError(f"tier must be 'default'/'large', got {tier!r}")


def _si_blind(canvas, px_m, lam, phases, kw, dbg):
    """solve_image 内部：单档盲解/锁相（不排序可用键，供两档复用）。"""
    try:
        if phases is not None:
            return solve_auto(canvas, lam, px_m, phases=phases, dbg=dbg, **kw)
        return solve_auto(canvas, lam, px_m, dbg=dbg, **kw)
    except Exception:
        if dbg:
            import traceback
            traceback.print_exc()
        return None


def _si_joint(canvas, px_m, kw, dbg):
    try:
        r = solve_auto_lambda(canvas, px_m, auto_center=kw.get(
            "auto_center", True), dbg=dbg, full=True)
        if r is None:
            return None
        # 2026-11-08:透传 σ(λ)——GUI 简并分级提示(σ≥2%=窄角域,
        # D/λ̂ 单独不可信;σ 小=宽角域联合解可信)。
        return (r["phase"], r["sol"], r["lam"], r.get("lam_sig"))
    except Exception:
        if dbg:
            import traceback
            traceback.print_exc()
        return None


def solve_image(img, px_m, lam=None, want="auto", phases=None, q_list=None,
                i_rel=None, dbg=False, solve_kw=None, fb_grid=None,
                mask=None):
    """R3 单一入口（2026-11-05，18-1104 需求）：画布选择(R1) → 相解 →
    零样本安全 fallback(R2) → 返回统一结果 dict。上层（zoo_run /
    phase_fit / gui_solver / ablation8）一律调本函数，不再自带画布/门控。

    参数
    ----
    img : 原生图（zoo 口径预处理后：哨兵≥1e9 与负值置零）
    px_m : **原生**像素尺寸(m)
    lam : 波长(Å)；None=联合 λ–D（solve_auto_lambda，此时 phases 必须为
        None——锁相联合未实现）
    want : 保留参数，现仅 "auto"
    phases : 相库子集（None=全库盲解；["X"]=锁相——评分空转，等价于
        solve_cascade+库内 I_rel）
    q_list : 锁表模式（phase_fit ceria 预注册 8 强环口径）：给定则直接
        solve_cascade(canvas, lam, px_m, q_list, i_rel, **kw)，绕过多相
        评分与 auto_center（须给 lam；返回 dict 的 phase=None）
    solve_kw : 透传 solve_auto / solve_cascade 的覆盖项，默认
        dict(cascade=True, auto_center=True, refine_2d=True)
        （锁表模式只取 rms_floor，其余走 solve_cascade 默认）
    fb_grid : (lo, hi, n) 覆盖救援网格（默认 FB_LAM_GRID；zoo 辅助模式
        传 (0.96, 1.04, 9) 锁相网格——T8 口径）

    返回 dict（None=两档均解不出）：
        phase(None=锁表模式), sol(画布坐标，含 2D 精化尾部若有),
        lam(求解实际用 λ：入参 / λ̂联合 / fallback 网格 λ̂), lam_in,
        canvas, s, px_m(画布), used_fallback, fb_reason
        (None | "a-none" | "b-rms-m" | "c-risk" | 同名+"-rejected"),
        ps, rms, m, def_ps/def_m/def_rms(默认档), lrg_ps/lrg_m/lrg_rms
        (升级档；未扫描为 None)——门控审计用

    R2 门控（零样本安全，不依赖 GT；取值依据 docs/unify_canvas_fallback.md）：
        (a) 默认档 None → 升级档 λ 网格救援，有效即采用；
        (b) 默认档 rms>FB_RMS_GATE 且 m<FB_M_GATE → 同 (a)，采用需 (ps,m,
            −rms) 键序严格优于默认档（T4 键序）；
        (c) 默认档 s≥FB_S_RISK 且非锁表模式 → **两段式**：先廉价分叉
            预检（大画布单 λ 经典解），两档 D 相对分叉 ≥FB_FORK 才上
            λ 网格救援，采用同 (b) 键序。
            ⚠ 18-1104 需求原案 (c) 为"ps 提升≥0.05 且 m 不降"，2026-11-05
            校准（probes/_r2_calib2.py）两头推翻：E4M/E9M 大档是同解重解
            （D 分叉 0.2%/0.07% 但 Δps +0.15/+0.25，ps 虚涨来自画布分辨
            率换算）会被误劫持；mar3450 救援解 m=5<Quartz 假解 m=11 会被
            误拒。D-分叉是唯一能同时分开两类的零样本判据（mar 108%）。
    """
    if want != "auto":
        raise ValueError(f"want={want!r} unsupported (only 'auto')")
    # B7 模块缝自动 mask(2026-11-07):行列暗带+P95 判据;与用户 mask OR。
    # _AUTO_GAP=False 可关(消融);Pilatus 全中零误报、Eiger 零动作、
    # Frelon 织构暗带全排除(探针 _b7_gapmask_probe.py 判决)。
    gap_n = 0
    if _AUTO_GAP:
        try:
            _g = auto_gap_mask(img)
            if _g is not None:
                gap_n = int(_g.sum())
                mask = _g if mask is None else (mask | _g)
        except Exception:
            gap_n = 0
    kw = dict(cascade=True, auto_center=True, refine_2d=True)
    if solve_kw:
        kw.update(solve_kw)
    lam_in = None if lam is None else _lam_A(lam)
    if q_list is not None and lam_in is None:
        raise ValueError("q_list 锁表模式必须给 lam")
    if lam_in is None and phases is not None:
        raise ValueError("锁相联合 λ–D 未实现（phases 需 lam）")

    def _solve_at(canvas, px_c, lam_eff):
        """单档求解 → (phase, sol, lam_used) 3 元组或 None。锁表模式
        phase=None、lam_used=lam_eff；联合模式 lam_used=λ̂。"""
        if q_list is not None:
            try:
                s = solve_cascade(canvas, lam_eff, px_c, q_list, i_rel,
                                  classic_first=True,
                                  rms_floor=kw.get("rms_floor", 3.5),
                                  dbg=dbg)
            except Exception:
                if dbg:
                    import traceback
                    traceback.print_exc()
                return None
            return (None, s, lam_eff) if s is not None else None
        if lam_eff is not None:
            r = _si_blind(canvas, px_c, lam_eff, phases, kw, dbg)
            return (r[0], r[1], lam_eff) if r is not None else None
        r = _si_joint(canvas, px_c, kw, dbg)
        return r if r is not None else None      # 已是 (ph, sol, λ̂)

    def _pack(res, canvas, s, px_c, used_fb, reason, dstat, lstat):
        return dict(phase=res[0], sol=res[1], lam=res[2], lam_in=lam_in,
                    lam_sig=(res[3] if len(res) > 3 else None),
                    canvas=canvas, s=s, px_m=px_c,
                    used_fallback=used_fb, fb_reason=reason,
                    ps=float(res[1][6]), rms=float(res[1][4]),
                    m=int(res[1][3]),
                    def_ps=dstat[0], def_m=dstat[1], def_rms=dstat[2],
                    lrg_ps=lstat[0], lrg_m=lstat[1], lrg_rms=lstat[2],
                    gap_masked=gap_n)   # B7:自动 mask 像素数(0=无)

    def _stat(res):
        return (None, None, None) if res is None else (
            float(res[1][6]), int(res[1][3]), float(res[1][4]))

    # ---- 默认档 ----
    c1, s1, _, _ = canvas_of(img, "default")
    if mask is not None:
        register_mask(mask, s1)
    r1 = _solve_at(c1, px_m * s1, lam_in)
    dstat = _stat(r1)

    # ---- 升级档要不要扫（R2） ----
    c2, s2, _, _ = canvas_of(img, "large")
    if mask is not None:
        register_mask(mask, s2)
    scan, reason = False, None
    if s2 != s1 and c2.shape[0] * c2.shape[1] <= FB_AREA_MAX:
        if r1 is None:
            scan, reason = True, "a-none"
        elif dstat[2] > FB_RMS_GATE and dstat[1] < FB_M_GATE:
            scan, reason = True, "b-rms-m"
        elif s1 >= FB_S_RISK and q_list is None:
            scan, reason = True, "c-risk"
    if not scan:
        return None if r1 is None else _pack(
            r1, c1, s1, px_m * s1, False, None, dstat, (None,) * 3)

    # ---- (c) 分叉预检（廉价）：大画布单 λ 解（cascade/refine_2d 关，
    # auto_center 必须开——E4M 束在画布一角，几何中心的经典解会出垃圾 D
    # → 假分叉 → 误劫持，2026-11-05 _r2_gate_test 实证）回答"大画布是否
    # 给出结构性不同的解"。同解重解（E4M/E9M：D 分叉 0.2%/0.07%，Δps
    # 虚涨到 +0.15/+0.25）不算画布档冲突，直接回默认档，省掉全质量网格
    # 扫描。分叉 ≥FB_FORK（mar 108%）才升级到 λ 网格救援。探针失败/环<3
    # 视为无分叉（大画布更没救）。----
    if reason == "c-risk":
        if lam_in is None:
            # 联合 λ–D 无同 λ 探针可言（λ 都未定）——直接保默认档
            return _pack(r1, c1, s1, px_m * s1, False, "c-risk-rejected",
                         dstat, (None,) * 3)
        probe_kw = dict(cascade=False, auto_center=True, refine_2d=False)
        rp = _si_blind(c2, px_m * s2, lam_in, phases, probe_kw, dbg)
        fork = None
        if rp is not None and rp[1][3] >= 3 and dstat[1] >= 3:
            fork = abs((rp[1][2] * s2) / (r1[1][2] * s1) - 1.0)
        if fork is None or fork < FB_FORK:
            return _pack(r1, c1, s1, px_m * s1, False,
                         "c-risk-rejected", dstat, _stat(rp))
        if dbg:
            print(f"  [fb] c-risk 分叉 {fork*100:.0f}% ≥ {FB_FORK*100:.0f}%"
                  f" → λ 网格救援")

    # ---- 救援扫描（a/b 或 c 分叉确认）：λ 网格 / 联合 ----
    if lam_in is not None:
        lo, hi, n = fb_grid if fb_grid else FB_LAM_GRID
        best = None
        for lg in np.geomspace(lo, hi, n) * lam_in:
            rr = _solve_at(c2, px_m * s2, float(lg))
            if rr is None:
                continue
            key = (rr[1][6], rr[1][3], -rr[1][4])
            if best is None or key > best[0]:
                best = (key, rr)
        r2 = None if best is None else best[1]
    else:
        r2 = _solve_at(c2, px_m * s2, None)
    lstat = _stat(r2)

    # ---- 采用判决：T4 键序 (ps, m, −rms) 严格优于默认档 ----
    # （校准：mar 救援解 m=5 < Quartz 假解 m=11，"m 不降"原案会误拒——
    # ps 首位比较先分胜负；E4M/E9M 同解重解已在分叉预检挡掉）
    adopt = r2 is not None and (
        True if reason == "a-none" else
        (lstat[0], lstat[1], -lstat[2]) > (dstat[0], dstat[1], -dstat[2]))
    if adopt:
        return _pack(r2, c2, s2, px_m * s2, True, reason, dstat, lstat)
    return None if r1 is None else _pack(
        r1, c1, s1, px_m * s1, False, reason + "-rejected", dstat, lstat)


_SH_CACHE = {}
_KASA_CACHE = {}


def _canvas_key(canvas):
    """缓存键用内容哈希，不用 id(canvas)。

    2026-11-02 修：id() 会被复用——52 帧批处理里均值画布可能撞上某帧的
    旧条目，拿到别人的中心（实测 alumina+r2d 出现 D=113.29/12 环，而单独
    跑同一张均值图是 118.32/7 环）。跨调用缓存必须内容寻址。
    """
    import hashlib
    arr = np.ascontiguousarray(canvas, dtype=np.float32)
    return (arr.shape,
            hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest())


def kasa_center(canvas, c0, n_rings=6):
    """相无关 Kasa 圆拟合定心（2026-11-01 W9）：粗中心径向谱最强 n 条
    完整环，带内(r±9px)高亮像素(>bg+0.5Δ) 2.5σ 剪枝迭代后代数圆拟合
    x²+y²=a·x+b·y+c，环间中位聚合。返回 (cx, cy, 残差σ中位)。

    实证（zoo 取证）：Eiger9M 从锐度中心 5.3→**1.1px**(σ0.18)，直入
    预言机证明的 2px 干净盆；Eiger4M 0.9σ0.16 / Pilatus2M 0.7σ0.14 无损。
    σ 即质检：好拟合 σ≤0.6，织构污染（mar3450）σ1.68 → 调用方以 σ<1.0
    为接受门。三条死路（锐度爬山/谐波定心/低峰数盆地）之后的首条活路。
    """
    H, W = canvas.shape
    peaks, prof, bp, good = ring_radii(canvas, c0[0], c0[1])
    if len(peaks) < 2:
        return (*c0, np.inf)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r_all = np.hypot(xx - c0[0], yy - c0[1])
    bg = float(np.percentile(canvas, 40))
    r_edge = min(c0[0], c0[1], W - 1 - c0[0], H - 1 - c0[1])
    centers, sigs = [], []
    for rp, h in sorted(peaks, key=lambda p: -p[1]):
        if len(centers) >= n_rings:
            break
        if not (40 < rp < 0.95 * r_edge):
            continue
        band = np.abs(r_all - rp) < 9.0
        iy, ix = np.nonzero(band)
        if iy.size < 200:
            continue
        v = canvas[iy, ix]
        keep = v > bg + 0.5 * (h - bg)
        if keep.sum() < 150:
            continue
        px_, py_ = ix[keep].astype(float), iy[keep].astype(float)
        cxk = cyk = sig = None
        for _ in range(3):
            A = np.stack([px_, py_, np.ones_like(px_)], axis=1)
            b = px_ ** 2 + py_ ** 2
            try:
                sol_, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                break
            a0, b0, cc = sol_
            cxk, cyk = a0 / 2.0, b0 / 2.0
            rk2 = cc + (a0 * a0 + b0 * b0) / 4.0
            if rk2 <= 0:
                break
            rk = np.sqrt(rk2)
            res = np.abs(np.hypot(px_ - cxk, py_ - cyk) - rk)
            sig = float(res.std())
            m = res < 2.5 * max(sig, 0.8)
            if m.sum() < 150 or m.all():
                break
            px_, py_ = px_[m], py_[m]
        if cxk is None:
            continue
        centers.append((cxk, cyk))
        sigs.append(sig)
    if not centers:
        return (*c0, np.inf)
    return (float(np.median([c[0] for c in centers])),
            float(np.median([c[1] for c in centers])),
            float(np.median(sigs)))


def center_by_sharpness(canvas, step=46, dbg=False):
    """谱锐度定心（2026-11-01 W9 层0修复）：多中心假设 → 各提径向谱 →
    按环衬度和择优。物理依据：真环心处全方位角环强度集中到同一半径 →
    谱峰高锐；错中心把环摊宽压矮。不依赖 RS 梯度（像板织构会骗 RS，
    mar3450 y 向飞 150px 实证），纯光度学证据。

    假设集：几何中心 + RS + 网格（step px，覆盖全画布）。
    打分：Σ(峰高−背景)/(背景+ε)，峰数做平手破除。
    结果按 canvas 缓存（solve_auto 逐相调用时复用）。"""
    key = _canvas_key(canvas)
    if key in _SH_CACHE:
        return _SH_CACHE[key]
    H, W = canvas.shape
    hypo = [((W - 1) / 2.0, (H - 1) / 2.0)]
    try:
        r = center_radial_symmetry(canvas)
        hypo.append(r)
    except Exception:
        pass
    yy = np.arange(step // 2, H, step, dtype=np.float64)
    xx = np.arange(step // 2, W, step, dtype=np.float64)
    for cy in yy:
        for cx in xx:
            hypo.append((cx, cy))
    best = (None, -np.inf, 0)
    for cx, cy in hypo:
        if not (10 < cx < W - 10 and 10 < cy < H - 10):
            continue
        try:
            peaks, prof, bp, good = ring_radii(canvas, cx, cy, nb=1024)
        except Exception:
            continue
        if len(peaks) < 3:
            continue
        bg = float(np.median(prof[prof > 0])) if (prof > 0).any() else 1.0
        score = sum(min((h - bg) / (bg + 1e-9), 60.0) for _, h in peaks)
        if score > best[1] + 1e-9 or \
                (abs(score - best[1]) < 1e-9 and len(peaks) > best[2]):
            best = ((cx, cy), score, len(peaks))
    out = best[0] if best[0] is not None else ((W - 1) / 2.0, (H - 1) / 2.0)
    _SH_CACHE[key] = out
    if dbg:
        print(f"    [sh] 锐度中心=({out[0]:.0f},{out[1]:.0f}) "
              f"geo=({(W-1)/2:.0f},{(H-1)/2:.0f}) score={best[1]:.1f}")
    return out


# ---- 环饥饿守卫(2026-11-06,issue #1 / gate_corners 标定)----
# classic-first 早退的可信度地板:D450 环饥饿域早退伪解的零样本指纹为
# ps<0(环序置换,−185~−251mm)或 m<4;干净域 ps≥0.477、m≥5
# (gate_sweep 69 + gate_corners 48 点),阈值留安全边。径向峰数不可作
# 判据:离轴/倾角涂抹反而多伪峰(_starve_calib 实证,d450_t2_o35 有
# 16 峰仍 3/3 早退错)。
# ---- v2(2026-11-06 晚,starve_guard.md "待改进"落地)----
# v1 的 EARLY_EXIT_PS=0.20 读 geo 候选的 ps——实测干净点 ps≈0.03 与
# 饥饿伪解(0.027~0.045)完全重叠,阈值等价"far 帧恒走共识":48 硬角
# 0/48 对,但 alumina 单帧基准 4.5/0.39→41/19(_starve_fp2 双口径标定)。
# v2 指纹 = RS 独立种子解与 geo 解的 D 偏离 |ln(D_rs/D_geo)|,仅 far 帧
# 计算(far=0 时锐度中心已投 geo 一票,层0守卫在,不付费):
#   must-fire(D450 饥饿族 11 点): 0.518~0.818
#   must-not-fire(alumina 54 帧): ≤0.406
# D_RS_GATE=0.46 取两类中点(余量 0.054/0.058)。误触发仅耗时代价
# (触发点共识解同样正确);RS 解失败=无反证,不触发。m 地板保留。
EARLY_EXIT_M = 4
D_RS_GATE = 0.46


def solve_cascade(canvas, lam, px_m, q_list, I_rel=None, *,
                  cx_nn=None, cy_nn=None, D_nn_m=None, center=None,
                  d_range=(0.05, 3.0), rms_floor=3.5,
                  rays=(1.0, 2.0, 3.5), classic_first=True,
                  dbg=False, trust_center=False):
    """F 架构：经典优先 + 多种子兜底（2026-10-31 生产化）。

    主路径（classic_first=True）：画布几何中心单种子经典解，m≥3 即返回。
    仅当主路径失败（None 或 m<3）才启用兜底：RS 定心 + 射线外推 + 可选
    NN 种子，按 (-m, -pscore, rms) 择优。classic_first=False 时为研究用的
    "全种子一起择优"（即 offset_duel5 的 F 臂）。

    为什么不做常开全种子：2026-10-31 ceria 实测，干净数据上射线伪解常以
    m=8 骗过 m-first 门（ΔD −111~−181mm），而真解 m=5~6；离轴域又必须靠
    RS/射线，且 pscore 在两个域都会误判（真解 ps 可低至 0，伪解可高至
    0.9）。两域共用同一标量门无解 → 生产采用"经典优先、失败才兜底"，
    把多标量门问题留给硬域实验（offset_duel5）。

    参数：cx_nn/cy_nn/D_nn_m —— NN 先验（可选，画布坐标/米制）。给了
    D_nn_m 时兜底 NN 种子走盆域门窄域 (0.35,2.8)，与生产旧路径一致。
    center=(cx,cy) —— 主路径/射线基准中心（默认画布几何中心）。
    返回 (cx,cy,D_px,m,rms,icorr,pscore,seed) 或 None；seed 为
    "geo"/"rs"/"ray1.0"/…/"nn"，供归因统计。
    """
    qs = np.asarray(q_list, float)
    I = (np.ones(len(qs)) if I_rel is None
         else np.asarray(I_rel, float))
    H, W = canvas.shape
    geo = ((W - 1) / 2.0, (H - 1) / 2.0) if center is None else center

    def run(cx, cy, D_ref, drange):
        peaks, prof, bin_px, good = ring_radii(canvas, cx, cy)
        return solve(peaks, cx, cy, 1.0 / px_m, qs, lam, px_m,
                     dbg=dbg, D_ref_m=D_ref, d_range=drange,
                     I_rel=I, prof_ctx=(prof, bin_px, good),
                     rms_floor=rms_floor, canvas=canvas)

    cands = []
    s_geo = run(geo[0], geo[1], 0.0, d_range)
    if trust_center:
        # 2026-11-01：调用方已给出可信中心（梯度投票，束在画布一角），
        # 不再让锐度中心/射线种子改写它——否则 46px 网格锐度中心会把
        # 好中心拖回 40px 外（Pilatus1M 实证）。
        if s_geo is not None and s_geo[3] >= 3:
            return s_geo + ("geo",)
        return None
    # 层0守卫（2026-11-01 W9）：谱锐度中心离 geo >15px（离轴证据）时，
    # geo 解不许早退——sh 是独立于 RS/geo 的光度学证据源，geo+sh 双票
    # 共识才能赢过各自单票的伪解（rays 全是 RS 衍生物不算独立票，
    # 修复3翻车教训）。锐度中心 ≈ geo（干净域）时快路径完全不变。
    sh = center_by_sharpness(canvas, dbg=dbg)
    # Kasa 精化（W9）：强环像素级圆拟合把锐度中心推到 ~1px（Eiger9M
    # 5.3→1.1px σ0.18，直入干净盆）。σ<1.0 质检门挡织构污染（mar3450
    # σ1.68），位移<30px 保险丝。结果缓存。
    kkey = _canvas_key(canvas)
    if kkey in _KASA_CACHE:
        sh, kasa_ok = _KASA_CACHE[kkey]
    else:
        kasa_ok = False
        try:
            ks = kasa_center(canvas, sh)
            if ks[2] < 1.0 and np.hypot(ks[0] - sh[0], ks[1] - sh[1]) < 30:
                sh = (ks[0], ks[1])
                kasa_ok = True
        except Exception:
            pass
        _KASA_CACHE[kkey] = (sh, kasa_ok)
    # 触发阈 8px（W9 kasa 后收紧；原 15px 会漏掉 Eiger9M 的 13px 案）。
    # alumina 全 51 帧 |sh−geo|≥233px（纹理拉偏）本就全在决斗分支，
    # 无新增暴露；ceria sh≈geo 不触发。
    far = float(np.hypot(sh[0] - geo[0], sh[1] - geo[1])) > 8.0
    # RS 独立种子(v2 守卫用;far 帧才解,结果留给兜底种子复用)
    rs_c = None
    s_rs = None
    if s_geo is not None and s_geo[3] >= 3:
        # 环饥饿守卫 v2(2026-11-06 晚):饥饿帧禁用全部早退分支(含
        # far 决斗),geo 落入候选走全种子+共识仲裁。指纹 = RS 独立解
        # 与 geo 解的 D 偏离(见文件头 v2 注释;_starve_fp2 标定:
        # 饥饿族 0.518~0.818 / alumina ≤0.406,门 0.46)。v1 的 ps
        # 阈值已废:干净点 geo ps≈0.03 与伪解重叠,等价恒触发,伤
        # alumina 单帧基准(41/19)。
        if far:
            rs_c = center_radial_symmetry(canvas)
            s_rs = run(rs_c[0], rs_c[1], 0.0, d_range)
        starved = (s_rs is not None
                   and abs(np.log(s_rs[2] / s_geo[2])) > D_RS_GATE) \
            or s_geo[3] < EARLY_EXIT_M

        if classic_first and not far and not starved:
            if dbg:
                print(f"    [cascade] geo 通过 m={s_geo[3]} "
                      f"rms={s_geo[4]:.2f} ps={s_geo[6]:+.3f} → 直接返回")
            return s_geo + ("geo",)
        if classic_first and far and not starved:
            # 决斗 v3（W9 kasa 证书）：kasa 拟合成功（σ<1.0，强环像素级
            # 圆拟合收敛）= 光度学证书，此时 sh 为主——geo 解必须严格优势
            # （ps>+0.25 且 rms<0.75×）才能翻盘。Eiger9M 案例：LaB6 在
            # geo 碎峰谱刷 m=11/cov10.5，在 kasa 干净谱塌到 ps0.103 →
            # 跨相仲裁砍掉，CeO2 真解胜。kasa 失败域（alumina 纹理帧
            # σ=inf、mar3450 σ1.68）自动退回 v2 决斗（geo 为主）。
            if kasa_ok:
                s_sh = run(sh[0], sh[1], 0.0, d_range)
                if s_sh is not None and s_sh[3] >= 4:
                    if dbg:
                        print(f"    [cascade] kasa证书 sh 为主 m={s_sh[3]} "
                              f"rms={s_sh[4]:.2f} ps={s_sh[6]:+.3f}")
                    return s_sh + ("sh",)
                if s_geo is not None and s_sh is not None and (
                        s_geo[6] > s_sh[6] + 0.25
                        and s_geo[4] < 0.75 * s_sh[4]):
                    if dbg:
                        print(f"    [cascade] geo 严格优势翻盘 "
                              f"m={s_geo[3]} rms={s_geo[4]:.2f}")
                    return s_geo + ("geo",)
                if s_sh is not None and s_sh[3] >= 3:
                    return s_sh + ("sh",)
                return s_geo + ("geo",)
            # 严格优势决斗 v2（2026-11-01 决斗取证）：sh 取代 geo 当且仅当
            # ps_sh > ps_geo+0.25 且 rms_sh < 0.75×rms_geo。m 不可作判据
            # （alumina 纹理帧 sh 伪解 m=12 刷过 geo 真解 m=9 而 D 错 60mm）；
            # Eiger4M 离轴帧 sh 真解 ps+0.35/rms4.6× 双条全过，四坏帧全挡。
            s_sh = run(sh[0], sh[1], 0.0, d_range)
            if s_sh is not None and s_sh[3] >= 3 and (
                    s_sh[6] > s_geo[6] + 0.25
                    and s_sh[4] < 0.75 * s_geo[4]):
                if dbg:
                    print(f"    [cascade] sh 严格优势 m={s_sh[3]} "
                          f"rms={s_sh[4]:.2f} ps={s_sh[6]:+.3f} vs geo "
                          f"m={s_geo[3]} rms={s_geo[4]:.2f} → 取代")
                return s_sh + ("sh",)
            return s_geo + ("geo",)
        cands.append(("geo", s_geo))
    if dbg and far:
        print(f"    [cascade] 锐度中心({sh[0]:.0f},{sh[1]:.0f})离 geo "
              f"{np.hypot(sh[0]-geo[0], sh[1]-geo[1]):.0f}px → 放行共识仲裁")
    if dbg and (s_geo is None or s_geo[3] < 3):
        print(f"    [cascade] 主路径{'失败' if s_geo is None else 'm<3'} "
              f"→ 兜底种子")

    rs = rs_c if rs_c is not None else center_radial_symmetry(canvas)
    seeds = [("rs", rs[0], rs[1]), ("sh", sh[0], sh[1])]
    for t in rays:
        seeds.append((f"ray{t}", rs[0] + t * (rs[0] - geo[0]),
                      rs[1] + t * (rs[1] - geo[1])))
    for tag, cx, cy in seeds:
        # v2 守卫已解过 RS 种子(far 帧)——原样复用,不重解
        if tag == "rs" and s_rs is not None:
            s = s_rs
        else:
            s = run(cx, cy, 0.0, d_range)
        if s is not None and s[3] >= 3:
            cands.append((tag, s))
            if dbg:
                print(f"    [cascade] {tag}: m={s[3]} rms={s[4]:.2f} "
                      f"ps={s[6]:+.3f} D={s[2] * px_m * 1000:.1f}mm")
    if cx_nn is not None and D_nn_m:
        s = run(cx_nn, cy_nn, D_nn_m, (0.35, 2.8))
        if s is not None and s[3] >= 3:
            cands.append(("nn", s))
            if dbg:
                print(f"    [cascade] nn: m={s[3]} rms={s[4]:.2f} "
                      f"ps={s[6]:+.3f} D={s[2] * px_m * 1000:.1f}mm")
    if not cands:
        return None
    # 兜底排序：跨种子共识（票数↓→pscore↓→m↓→rms↑）。
    # 2026-10-31 两域判决（cascade_probe 复判 + offset_duel5 G 臂）：
    # 干净域 m-first 会选射线伪解（3/7 档 −110~−181mm 灾难），共识门
    # 0 灾难（真解 geo+nn 天然双票，伪解各说各话单票）；离轴域共识
    # 与 m-first 持平（27% vs 27%，近箱中位更优）→ 兜底换共识。
    cp = consensus_pick(cands)
    tag, s = cp[0], cp[1]
    if dbg:
        print(f"    [cascade] 共识胜者 {tag}（{cp[2]} 票）")
    return s + (tag,)



# ==================== B13 多帧联合标定(2026-11-08) ====================
# 判决依据 probes/_b13_joint.py(51 帧):σ(D_joint)=0.010mm=MAD/√51
# ✓√N 收敛;逐帧 MAD 0.084mm(控制点口径)。两层设计:GUI 批量与
# CLI 探针复用同代码(上层已有逐帧解时跳过重解,只补控制点+联合)。

def cpts_of_frame(img, cx, cy, D_px, px1, px2, lam, qs, nsec=36,
                  mass_rel=0.10, win=6.0, bump=1.5, fwd=None):
    """原生图 + 圆解种子(原生 px)→ 控制点(B13 联合最小单元)。

    种子容差:画布盲解换算即可(D ±2% 内、中心 ±(s/2+若干)px;
    fit_tilt 内部 sector 峰定位吸收)。fwd=None → 自研前向;
    给 make_pyfai_fwd(px1,px2,spline) → spline 口径(装置级对齐
    GT poni,如 Frelon distorsion_2x2)。返回 dict(d1,d2,tth,w,
    p1,p2, rms5, D) 或 None(<20 点/拟合失败)。
    """
    from poni_out import fit_tilt
    do = {}
    tf = fit_tilt(img, float(cx), float(cy), float(D_px), qs, lam,
                  px1, px2, sfull=1, nsec=nsec, win=win, mass_rel=mass_rel,
                  bump=bump, fwd=fwd, data_out=do)
    if tf is None or len(do.get("d1", ())) < 20:
        return None
    return dict(d1=np.asarray(do["d1"], float),
                d2=np.asarray(do["d2"], float),
                tth=np.asarray(do["tth"], float),
                w=np.asarray(do["w"], float),
                p1=tf["poni1"], p2=tf["poni2"], rms5=tf.get("rms5"),
                D=tf["dist"] * 1e3)      # 帧独立 5 参解 D(mm,审计用)


def b_route_mean_seed(imgs, px1, px2, lam, phase, spline=None):
    """B 路线 mean 种子(phase_fit alumina Phase3 提取,2026-11-08)。

    弱衍射静态装置专用:逐帧盲解次优盆丛生(alumina 5/8 帧 −2~−4mm,
    ps/m/rms 无区分度),52 帧平均把峰位噪声 ↓√N 后一次圆解即可靠。
    链:流式 mean → solve_image 画布圆解(锁相+旧三关)→ fit_tilt
    (画布 + spline 前向,nsec=36)→ 5 参种子(米制)。
    imgs: 原生图迭代器(调用方负责 load 与哨兵清洗)。
    返回 dict(dist,poni1,poni2,rot1,rot2,n) 或 None。
    种子→原生 px:cx=poni2/px2−0.5, cy=poni1/px1−0.5, D_px=dist/px1。
    """
    from poni_out import fit_tilt, make_pyfai_fwd
    acc, n = None, 0
    for img in imgs:
        img = np.where(np.isfinite(img), img, 0.0).astype(np.float64)
        acc = img if acc is None else acc + img
        n += 1
    if n < 1:
        return None
    mean_img = acc / n
    px_avg = (px1 + px2) / 2
    r = solve_image(mean_img, px_avg, lam=lam, phases=[phase],
                    solve_kw=dict(cascade=True, auto_center=False,
                                  refine_2d=False))
    if r is None:
        return None
    canvas, s, sol = r["canvas"], r["s"], r["sol"]
    tf = fit_tilt(canvas, sol[0], sol[1], sol[2], phase_qs(phase), lam,
                  px1 * s, px2 * s, sfull=s,
                  fwd=make_pyfai_fwd(px1, px2, spline), nsec=36,
                  mass_rel=0.10, bump=1.5)
    if tf is None:
        return None
    return dict(dist=tf["dist"], poni1=tf["poni1"], poni2=tf["poni2"],
                rot1=tf["rot1"], rot2=tf["rot2"], n=n, rms5=tf.get("rms5"))


def joint_solve_frames(cpts_list, px1, px2, boot=6, seed=3, spline=None):
    """控制点级联合拟合:全局 (dist,rot1,rot2) + 每帧 (poni1,poni2)。

    cpts_list: cpts_of_frame 输出(≥2 帧有效才有意义;≥3 起用)。
    spline: 探测器畸变文件路径(None=自研前向)。**口径一致性**:
    cpts 用 fwd=spline 提取时,联合必须同 spline,否则控制点与
    前向不匹配(系统差 ~0.6mm,见 B13 无 spline Δ−0.588)。
    boot: bootstrap 抽帧(70%)次数,≥1 时返回 sigma_boot。
    返回 dict(dist, rot1, rot2, lam 无, p1s/p2s 每帧, rms(rad),
    npts, nf, sigma_boot|None, Ds_boot) 或 None。
    """
    from poni_out import make_pyfai_fwd
    from scipy.optimize import least_squares
    live = [c for c in cpts_list if c is not None]
    if len(live) < 2:
        return None
    fwd = make_pyfai_fwd(px1, px2, spline)
    D1 = np.concatenate([c["d1"] for c in live])
    D2 = np.concatenate([c["d2"] for c in live])
    TT = np.concatenate([c["tth"] for c in live])
    WW = np.concatenate([c["w"] for c in live])
    FR = np.concatenate([[k] * len(c["d1"]) for k, c in enumerate(live)])
    WW = WW / WW.mean()

    def resid(p):
        return (fwd(p[0], p[3 + 2 * FR], p[4 + 2 * FR], p[1], p[2], D1, D2)
                - TT) * WW

    # 初值:每帧 p1/p2(fit_tilt 解),全局 dist 一阶粗估、rot=0
    i0 = int(np.argmax(TT))
    r0 = float(np.hypot((D1[i0] + 0.5) * px1, (D2[i0] + 0.5) * px2))
    dist0 = r0 / np.tan(TT[i0])
    p0 = np.concatenate([[dist0, 0.0, 0.0],
                         np.array([[c["p1"], c["p2"]]
                                   for c in live]).ravel()])
    try:
        fit = least_squares(resid, p0, method="lm", max_nfev=4000)
    except Exception:
        return None
    rms = float(np.sqrt(np.mean((fit.fun / WW) ** 2)))

    out = dict(dist=float(fit.x[0]), rot1=float(fit.x[1]),
               rot2=float(fit.x[2]), rms=rms,
               npts=int(len(TT)), nf=len(live),
               p1s=[float(v) for v in fit.x[3::2]],
               p2s=[float(v) for v in fit.x[4::2]])
    if boot >= 1:
        rng = np.random.default_rng(seed)
        nf = len(live)
        Ds_b = []
        for _ in range(boot):
            sel = rng.choice(nf, size=max(3, int(nf * 0.7)), replace=False)
            mask = np.isin(FR, sel)
            remap = {v: k for k, v in enumerate(sorted(set(sel.tolist())))}
            FRb = np.array([remap[v] for v in FR[mask]])
            D1b, D2b, TTb, WWb = (D1[mask], D2[mask], TT[mask], WW[mask])

            def rb(pp):
                return (fwd(pp[0], pp[3 + 2 * FRb], pp[4 + 2 * FRb],
                            pp[1], pp[2], D1b, D2b) - TTb) * WWb

            p0b = np.concatenate([[fit.x[0], fit.x[1], fit.x[2]],
                                  np.array([[c["p1"], c["p2"]]
                                            for k, c in enumerate(live)
                                            if k in set(sel.tolist())]
                                           ).ravel()])
            try:
                fb = least_squares(rb, p0b, method="lm", max_nfev=2500)
                Ds_b.append(fb.x[0] * 1e3)
            except Exception:
                continue
        out["Ds_boot"] = Ds_b
        out["sigma_boot"] = (float(np.std(Ds_b))
                             if len(Ds_b) >= 3 else None)
    return out
