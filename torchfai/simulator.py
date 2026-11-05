# -*- coding: utf-8 -*-
"""正向仿真器：给定随机几何 → pyFAI q-map → 真实标样德拜环合成 → 扰动 → uint16 图。

设计要点
--------
1. q-map 用 pyFAI Geometry.array_from_unit 逐样本计算 —— 与 .poni 真值语义完全一致
   （环半径 vs 解析解已在 probe 中 1e-5 对齐）。
2. 谱 = Σ Lorentzian(真实标样 d-spacings) + 背景 + 可选非晶驼峰；Poisson 噪声（光子计数）。
3. 扰动：热点/坏点/束流挡光圆盘/Pilatus 模间隙死区（原生 mask 裁偶 rebin）。
4. 有效性规则：可见环 < 2 条则重采几何（避免无信息样本进入训练集，已记录在案）。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pyFAI.detectors import detector_factory
from pyFAI.geometry import Geometry

from .config import SimConfig
from .phases import PHASE_NAMES, phase_qi


# ---------------------------------------------------------------- 探测器束
@dataclass
class DetectorBundle:
    """进程内单例：binned 探测器 + 死区 + 坐标网格。"""
    name: str
    shape: tuple              # (H, W)
    pixel: float              # m（正方形像素）
    dead: np.ndarray          # bool (H, W)，模间隙等
    pyfai_det: object         # pyFAI Detector（已设 binning），建 Geometry 时必须挂上
    yy: np.ndarray            # 行索引网格
    xx: np.ndarray            # 列索引网格

    @property
    def extent(self) -> tuple:
        return self.shape[0] * self.pixel, self.shape[1] * self.pixel  # (H_m, W_m)


@lru_cache(maxsize=4)
def get_detector(name: str, binning: int) -> DetectorBundle:
    det = detector_factory(name)
    mask_native = det.mask.copy()          # 必须在 set binning 前取（奇数维 rebin 会炸）
    det.binning = (binning, binning)
    H, W = det.shape                       # 已是 binned 维度 (521, 490)
    Hn, Wn = H * binning, W * binning      # 原生被覆盖的偶数维：1042, 980
    dead = mask_native[:Hn, :Wn].reshape(H, binning, W, binning).any(axis=(1, 3))
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return DetectorBundle(name, (H, W), det.pixel1, dead, det, yy, xx)


_SCALED_DB_CACHE: dict[tuple, "DetectorBundle"] = {}


def get_scaled_detector(name: str, binning: int, scale: float) -> DetectorBundle:
    """像素尺度 ×scale 的探测器束（10-29 视域域扩展）。

    网格形状/死区样式不变，像素 = 基准×scale → 视域 = 基准×scale。
    物理语义：同款模块布局的大像素（或等效远置）探测器。环半径/环宽按
    真实像素粒度走 q-map + 泊松（不是事后缩放图像——噪声统计才对）。
    scale=1 时与 get_detector 逐位一致。feasible_dist_loggrid 缓存键含
    db.pixel，不同 k 自动分缓存。
    """
    key = (name, binning, round(float(scale), 4))
    hit = _SCALED_DB_CACHE.get(key)
    if hit is not None:
        return hit
    det = detector_factory(name)
    mask_native = det.mask.copy()
    det.binning = (binning, binning)
    H, W = det.shape
    Hn, Wn = H * binning, W * binning
    dead = mask_native[:Hn, :Wn].reshape(H, binning, W, binning).any(axis=(1, 3))
    det.pixel1 = det.pixel1 * key[2]
    det.pixel2 = det.pixel2 * key[2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    db = DetectorBundle(f"{name}x{key[2]:g}", (H, W), det.pixel1, dead, det, yy, xx)
    _SCALED_DB_CACHE[key] = db
    return db


# ---------------------------------------------------------------- 采样
def sample_geometry(rng: np.random.Generator, cfg: SimConfig, db: DetectorBundle,
                    feas_lg: np.ndarray | None = None) -> dict:
    """6 几何参数随机采样。dist 用 log-uniform；给定可行域时仅在相可行区间内采。"""
    if feas_lg is not None and len(feas_lg) >= 2:
        dlo, dhi = float(feas_lg[0]), float(feas_lg[-1])
    else:
        dlo, dhi = np.log10(cfg.dist_range[0]), np.log10(cfg.dist_range[1])
    dist = float(10.0 ** rng.uniform(dlo, dhi))
    Hm, Wm = db.extent
    lo, hi = cfg.poni_frac_range
    poni1 = float(rng.uniform(lo, hi) * Hm)
    poni2 = float(rng.uniform(lo, hi) * Wm)
    rot1 = float(rng.uniform(*cfg.rot1_range))
    rot2 = float(rng.uniform(*cfg.rot2_range))
    rot3 = float(rng.uniform(*cfg.rot3_range))
    return dict(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3)


def make_geometry(p: dict, cfg: SimConfig, db: DetectorBundle,
                  wavelength_m: float | None = None) -> Geometry:
    """Geometry 必须挂 binned detector（默认 detector 的 pixel=None 会炸）。"""
    return Geometry(detector=db.pyfai_det, dist=p["dist"], poni1=p["poni1"], poni2=p["poni2"],
                    rot1=p["rot1"], rot2=p["rot2"], rot3=p["rot3"],
                    wavelength=cfg.wavelength if wavelength_m is None else wavelength_m)


def ring_radius_m(q: np.ndarray, dist: float, lam: float) -> np.ndarray:
    """未倾斜近似下，q -> 探测器上的环半径 r = L·tan(2θ)。仅用于可见性统计。"""
    two_theta = 2.0 * np.arcsin(np.clip(q * lam / (4.0 * np.pi), 0.0, 0.999999))
    return dist * np.tan(two_theta)


# ---------------------------------------------------------------- 可行域剪枝
_FEAS_CACHE: dict[tuple, np.ndarray] = {}


def feasible_dist_loggrid(ph: str, db: "DetectorBundle", cfg: SimConfig,
                          wavelength_m: float | None = None) -> np.ndarray:
    """该相在当前探测器/配置下"至少 min_visible_peaks 条环可见"的 log10(dist) 网格点。

    保守假设：最大挡光盘 + poni 居中（r_far 最小），保证采样后精判几乎必过。
    可见域对 dist 近似单区间（太近→环落入束流孔，太远→环出界），取首尾即可。
    wavelength_m：λ-conditioned 卷的每样本波长（离散网格档传入，缓存友好）。
    """
    lam_m = cfg.wavelength if wavelength_m is None else wavelength_m
    key = (ph, db.name, db.shape, round(db.pixel, 9), round(lam_m, 13),
           cfg.min_visible_peaks, cfg.dist_range, max(cfg.beamstop_r_range))
    hit = _FEAS_CACHE.get(key)
    if hit is not None:
        return hit
    qs, _ = phase_qi(ph)
    lam_A = lam_m * 1e10
    Hm, Wm = db.extent
    r_far = 1.05 * float(np.hypot(0.75 * Hm, 0.75 * Wm))           # 中位 poni 情形；极端由内层重采兜底
    thr_lo = (max(cfg.beamstop_r_range) + 3.0) * db.pixel          # 最大挡光盘保守
    lg = np.linspace(np.log10(cfg.dist_range[0]), np.log10(cfg.dist_range[1]), 241)
    ok = []
    for v in lg:
        rr = ring_radius_m(qs, 10.0 ** v, lam_A)
        if ((rr > thr_lo) & (rr < r_far)).sum() >= cfg.min_visible_peaks:
            ok.append(v)
    arr = np.asarray(ok, dtype=np.float64)
    _FEAS_CACHE[key] = arr
    return arr


# ---------------------------------------------------------------- 合成
def synth_sample(rng: np.random.Generator, cfg: SimConfig, db: DetectorBundle,
                 phase_idx: int | None = None,
                 geom: dict | None = None,
                 pixel_scale: float = 1.0) -> dict:
    """生成一张样本。返回 dict(img u16, params f32[6], aux f32[7], n_reject)。

    aux: [phase_idx, n_vis, qmax, scale, bs_r, lam_A, px_scale]（第 6 位波长 Å，
    第 7 位像素尺度 k=像素/基准像素；旧 6 列卷无此位——读端缺省 k=1）。
    pixel_scale 仅记录标签；几何真值来自传入的 db（生成端配
    get_scaled_detector 保证一致）。宽峰分层：每样本以 broad_frac 概率
    落宽峰 regime（环宽在 broad_range log-uniform），否则旧窄峰域。
    """
    # ---- 波长决策：固定 or λ-conditioned 网格档(log 域随机) + 抖动 ----
    if cfg.wavelength_range is not None:
        lo_A, hi_A = cfg.wavelength_range[0] * 1e10, cfg.wavelength_range[1] * 1e10
        grids = np.exp(np.linspace(np.log(lo_A), np.log(hi_A),
                                   int(round((hi_A - lo_A) / 0.05)) + 1))   # log 域 0.05Å 步
        lam_grid_m = float(grids[rng.integers(len(grids))]) * 1e-10          # 网格档（feas 缓存键）
        lam_m = lam_grid_m + rng.uniform(-0.01, 0.01) * 1e-10                # 抖动 ±0.01Å（渲染/标签）
    else:
        lam_grid_m = cfg.wavelength
        lam_m = cfg.wavelength
    lam_A = lam_m * 1e10
    rejects = 0
    success = False
    for _phase_attempt in range(8):          # 外层：换相 ≤8 次（fix: phase 偏斜）
        ph = PHASE_NAMES[rng.integers(len(PHASE_NAMES))] if phase_idx is None \
            else PHASE_NAMES[phase_idx]
        feas = None if geom is not None else feasible_dist_loggrid(ph, db, cfg,
                                                                   wavelength_m=lam_grid_m)
        if feas is not None and len(feas) < 2:
            continue                          # 该相在此探测器/配置下无可行 dist，换相
        for _geo_attempt in range(31):        # 内层：固定相重采几何 ≤30 次
            g_dict = geom if geom is not None else sample_geometry(rng, cfg, db, feas)

            # ---- 可见环统计（未倾斜近似，粗判即可）----
            qs, Is = phase_qi(ph)
            # 束流挡光盘
            if rng.random() < cfg.beamstop_prob:
                bs_r = float(rng.uniform(*cfg.beamstop_r_range))
            else:
                bs_r = -1.0
            rr = ring_radius_m(qs, g_dict["dist"], lam_A)
            # 探测器上离 poni 最远的角距离
            Hm, Wm = db.extent
            corners = np.array([g_dict["poni1"], Hm - g_dict["poni1"],
                                g_dict["poni2"], Wm - g_dict["poni2"]])
            r_far = float(np.hypot(max(corners[0], corners[1]), max(corners[2], corners[3])))
            thr_lo = (bs_r + 3.0) * db.pixel if bs_r > 0 else 0.0
            vis = (rr > thr_lo) & (rr < 1.05 * r_far)
            if geom is not None or vis.sum() >= cfg.min_visible_peaks:
                success = True
                break
            rejects += 1
        if success:
            break
    if not success:
        raise RuntimeError("no feasible phase/geometry combination after resampling")

    # ---- q map ----
    g = make_geometry(g_dict, cfg, db, wavelength_m=lam_m)
    q = g.array_from_unit(shape=db.shape, typ="center", unit="q_A^-1", scale=True)
    q = np.nan_to_num(q, nan=100.0, posinf=100.0)

    # ---- 谱（相对单位）----
    img = np.full(q.shape, rng.uniform(*cfg.bg0_range), dtype=np.float64)
    img += rng.uniform(*cfg.bg1_range) * q
    if rng.random() < cfg.hump_prob:
        qh = rng.uniform(*cfg.hump_q_range)
        sh = rng.uniform(*cfg.hump_sigma_range)
        ah = rng.uniform(*cfg.hump_amp_range)
        img += ah * np.exp(-0.5 * ((q - qh) / sh) ** 2)

    B = rng.uniform(*cfg.dw_b_range)
    # Per-ring σq record (v4s2 replay labeling): only appends already-drawn local variables,
    # zero new random calls → random stream byte-identical, replay-safe
    sigs = []
    # Broad-peak stratification decision (per sample; when broad_frac=0 no random number is drawn—old config RNG stream bitwise unchanged)
    is_broad = (cfg.peak_sigma_broad_frac > 0.0
                and rng.random() < cfg.peak_sigma_broad_frac)
    if is_broad:
        s_lo, s_hi = cfg.peak_sigma_broad_range
        ln_lo, ln_hi = np.log(s_lo), np.log(s_hi)
    else:
        s_lo, s_hi = cfg.peak_sigma_range
    ring_max_vis = 0.0     # 最大峰高（用于绝对标度）
    ring_max_all = 0.0
    for q0, I0, v in zip(qs, Is, vis):
        A = I0 * rng.uniform(*cfg.amp_jitter) * float(np.exp(-B * q0 * q0))
        if is_broad:
            sig = float(np.exp(rng.uniform(ln_lo, ln_hi)))   # 宽峰 log-uniform
        else:
            sig = rng.uniform(s_lo, s_hi)
        sigs.append(sig) if v else None   # 只记可见环（网络实际看到的）
        img += A * sig * sig / ((q - q0) ** 2 + sig * sig)   # Lorentzian，峰高 A
        ring_max_all = max(ring_max_all, A)
        if v:
            ring_max_vis = max(ring_max_vis, A)
    n_vis = int(vis.sum())

    # ---- 绝对标度：最强可见环峰高 → 目标计数（无可见环时退用全谱最强）----
    if getattr(cfg, "peak_counts_loguniform", False):
        lo_c, hi_c = cfg.peak_counts_range
        target = float(np.exp(rng.uniform(np.log(lo_c), np.log(hi_c))))
    else:
        target = rng.uniform(*cfg.peak_counts_range)
    scale = target / max(ring_max_vis if ring_max_vis > 0 else ring_max_all, 1e-9)
    img *= scale

    # ---- 光子统计 + 读出 ----
    img = rng.poisson(img).astype(np.float32)

    # ---- 扰动：热点 / 随机坏点 ----
    imax = float(img.max())
    hot_rate = rng.uniform(*cfg.hot_rate_range)
    if hot_rate > 0:
        n_hot = int(rng.poisson(hot_rate * img.size))
        if n_hot:
            idx = rng.integers(0, img.size, n_hot)
            img.ravel()[idx] = rng.uniform(*cfg.hot_val_range) * imax
    dead_rate = rng.uniform(*cfg.dead_rate_range)
    if dead_rate > 0:
        n_dead = int(rng.poisson(dead_rate * img.size))
        if n_dead:
            idx = rng.integers(0, img.size, n_dead)
            img.ravel()[idx] = 0.0

    # ---- 束流挡光圆盘（中心=poni 像素位置）----
    if bs_r > 0:
        rc, cc = g_dict["poni1"] / db.pixel, g_dict["poni2"] / db.pixel
        m = (db.yy - rc) ** 2 + (db.xx - cc) ** 2 <= bs_r * bs_r
        img[m] = 0.0

    # ---- 模间隙死区 + 饱和 ----
    img[db.dead] = 0.0
    np.clip(img, 0, cfg.sat_clip, out=img)
    img_u16 = img.astype(np.uint16)

    params = np.array([g_dict["dist"], g_dict["poni1"], g_dict["poni2"],
                       g_dict["rot1"], g_dict["rot2"], g_dict["rot3"]], np.float32)
    aux = np.array([float(PHASE_NAMES.index(ph)), float(n_vis),
                    float(np.nanmax(q)), float(scale), bs_r, lam_A,
                    float(pixel_scale)], np.float32)
    return dict(img=img_u16, params=params, aux=aux, n_reject=rejects,
                phase=ph, geom=g_dict,
                sigmas=np.asarray(sigs, np.float32))
