# -*- coding: utf-8 -*-
"""仿真配置：所有采样范围/扰动参数集中于此（单一事实来源，落盘进 meta.json）。"""
from dataclasses import dataclass, asdict, field


@dataclass
class SimConfig:
    # ---- 探测器 ----
    detector_name: str = "Pilatus_1M"
    binning: int = 2                  # (1043,981) -> (521,490), pixel 344 µm

    # ---- 波长（v1 固定 Cu Kα1；真实束线波长已知，不是校准未知量）----
    wavelength: float = 1.5406e-10    # m
    # λ-conditioned 训练卷用：None=固定上面的 wavelength；否则 (lo, hi) m，
    # 每样本按 0.05Å 网格档 log 域随机（可行域缓存友好），渲染用 ±0.01Å 抖动连续值
    wavelength_range: tuple | None = None

    # ---- 像素尺度条件化（10-29 覆盖洞修复：视域域扩展）----
    # None=固定 k=1（旧行为，逐位兼容）；生成端传离散 k 档列表逐样本随机，
    # 每样本 k 记入 aux 第 7 列（无量纲，像素 = 基准×k，视域 = 基准×k）
    pixel_scale_range: tuple | None = None

    # ---- 6 个几何参数的采样范围 ----
    dist_range: tuple = (0.10, 3.0)        # m，log-uniform；下限 0.10 保证高 q 相第二环可见（WAXS 近几何）
    poni_frac_range: tuple = (0.06, 0.94)  # poni ∈ [lo,hi]×探测器边长
    rot1_range: tuple = (-0.07, 0.07)      # rad ≈ ±4°
    rot2_range: tuple = (-0.07, 0.07)
    rot3_range: tuple = (-0.17, 0.17)      # rad ≈ ±10°（rot3 对环位置最不敏感，范围收窄保可辨识性）

    # ---- 散射谱 ----
    peak_sigma_range: tuple = (0.002, 0.012)   # Å⁻¹，Lorentzian 半宽
    # 宽峰分层（10-29 alumina 判决：真实宽峰标样 σq≈0.15-0.6，旧域 12-50 倍出界）：
    # 每样本以 broad_frac 概率落入宽峰 regime，环宽在 broad_range 内 log-uniform
    peak_sigma_broad_frac: float = 0.0
    peak_sigma_broad_range: tuple = (0.05, 0.6)
    amp_jitter: tuple = (0.5, 1.6)             # 每峰强度相对扰动
    dw_b_range: tuple = (0.0, 0.02)            # Debye-Waller 式衰减 exp(-B q²)
    bg0_range: tuple = (0.0, 0.03)             # 常数背景
    bg1_range: tuple = (0.0, 0.02)             # q 线性背景系数
    hump_prob: float = 0.35                    # 非晶驼峰出现概率
    hump_q_range: tuple = (0.8, 2.2)
    hump_sigma_range: tuple = (0.15, 0.4)
    hump_amp_range: tuple = (0.1, 0.5)
    peak_counts_range: tuple = (3000.0, 30000.0)  # 最强可见环的目标峰值计数
    peak_counts_loguniform: bool = False   # True: log 域均匀采样曝光（跨数量级域扩展，
                                           # 如 (30,30000) 覆盖真实短曝光 regime；10-28 域差）

    # ---- 扰动 ----
    hot_rate_range: tuple = (0.0, 3e-4)   # 热点像素比例
    hot_val_range: tuple = (0.5, 3.0)     # 热点值 × 图像最大值
    dead_rate_range: tuple = (0.0, 1e-4)  # 随机坏点比例
    beamstop_prob: float = 0.7
    beamstop_r_range: tuple = (8.0, 45.0) # binned px
    sat_clip: int = 65535                 # uint16 饱和

    # ---- 样本有效性 ----
    min_visible_peaks: int = 2   # 至少 2 条可见环，否则重采几何（保持训练样本信息量）

    # ---- 存储 ----
    shard_size: int = 256

    def to_dict(self):
        return asdict(self)
