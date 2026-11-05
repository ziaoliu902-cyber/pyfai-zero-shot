# -*- coding: utf-8 -*-
"""solver_core.py：相库驱动的通用峰位几何解算器（B 路线核心库）。

2026-10-31 从 ring_fit.py（ceria 专用脚本）泛化：
- 相库驱动：q 表来自 torchfai.phases（CeO2/LaB6/Si/Al2O3/Quartz/AgBh）
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
    v = canvas.astype(np.float64).ravel()
    cnt = np.bincount(idx, minlength=nb)
    prof = np.bincount(idx, weights=v, minlength=nb) / np.maximum(cnt, 1)
    good = cnt > 0.15 * np.median(cnt[cnt > 0])
    bg = float(np.median(prof[cnt > 0]))
    pk, _ = find_peaks(prof, prominence=prom * prof.max(), distance=12)
    out = []
    bin_px = rmax / nb
    for p in pk:
        if p <= 2 or p >= nb - 3:
            continue
        if prof[p] <= bg:            # 角落涟漪/负向抖动，非衍射峰
            continue
        y0, y1, y2 = prof[p - 1], prof[p], prof[p + 1]
        d = 0.5 * (y0 - y2) / max(y0 - 2 * y1 + y2, 1e-12)
        rp = (p + 0.5 + np.clip(d, -1, 1)) * bin_px
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
    # 种子 = 投票簇 top-3（实证最优；网格全局扫描已证伪——宽先验喂出
    # 0.63×/1.1× 缩放伪解，2026-10-31 alumina）
    clusters = [c for c in clusters if c[0] >= 3][:3]
    if not clusters:
        if dbg:
            print("    [dbg] 无≥3票簇")
        return None
    seeds = [float(D) for _, D in clusters]

    def pair_up(cx, cy, D, frac):
        pairs = []
        for qi, q in enumerate(qs):
            rm = _r_model(np.array([q]), lam, D)[0]
            tol = max(1.5, frac * rm)
            c = [(abs(rp - rm), rp, h) for rp, h in peaks_rp
                 if abs(rp - rm) < tol]
            if c:
                _, rp, h = min(c)
                pairs.append((q, rp, h, I_rel[qi]))
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


def solve_auto(canvas, lam, px_m, phases=None, dbg=False,
               D_ref_m=0.0, cx0=None, cy0=None, d_range=(0.05, 3.0),
               rms_floor=3.5):
    """多相自动识别（全托管）：画布 → 峰 + 径向谱 → 逐相 solve →
    按【全谱相关 pscore ↓ 覆盖分 ↓ rms ↑】全局择优。
    pscore：合成谱 vs 实测谱 Pearson 相关——伪相/索引错位解只碰得上
    峰位置子集，实测谱在其"预测空档"处的真环令相关崩盘（2026-10-31）。
    返回 (phase_name, sol) 或 None，sol=(cx,cy,D_px,m,rms,icorr,pscore)。"""
    from torchfai.phases import PHASES
    H, W = canvas.shape
    if cx0 is None:
        cx0 = (W - 1) / 2.0
    if cy0 is None:
        cy0 = (H - 1) / 2.0
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
        cands.append((cov, s[6], s[3], -s[4], ph, s))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][4], cands[0][5]
