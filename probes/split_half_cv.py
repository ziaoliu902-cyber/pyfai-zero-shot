# -*- coding: utf-8 -*-
"""split_half_cv.py：少环（m <= 5）拆分交叉验证模块 (Split-Half Cross-Validation)

物理与数学依据：
  当衍射可见环较少（m <= 5 或高能/短距离场景）时，反问题约束减少，
  传统单目标拟合容易通过非物理的中心/距离代偿，将错误物相或错配环序强行拟合成
  残差极小的“自洽伪解”（假盆地陷阱，False Basin）。

拆分交叉验证原理：
  1. 将检出的 m 条环按序分成互斥的两组：
     - 训练集 Set A（偶数序环：0, 2, 4, ...）
     - 验证集 Set B（奇数序环：1, 3, 5, ...）
  2. 仅用 Set A 拟合几何参数 Θ_A = (D_A)；
  3. 用 Θ_A 外推预测 Set B 的环半径并计算泛化残差 RMS_val；
  4. 比较验证残差与训练残差之比：
     cv_ratio = RMS_val / max(RMS_train, 0.2 px)
  5. 物理判据：
     - 若为真实物理盆地：外推残差自然极低，cv_ratio ≈ 1.0 (< 1.8)，通过验证；
     - 若为强行拟合的伪解：Set A 的代偿无法外推至 Set B，cv_ratio >> 2.5，触发伪解拦截！
"""

import numpy as np


def split_half_verify(ring_radii_px, qs, lam, D_px=None, max_ratio=2.5,
                      max_val_rms_px=2.0):
    """对一组匹配环执行拆分交叉验证。

    参数：
      ring_radii_px: 检出衍射环半径列表（像素）
      qs: 对应的倒易矢量模 q 列表（Å⁻¹）
      lam: X 射线波长（Å）
      D_px: 可选，全局初值距离（像素）
      max_ratio: 容许的外推残差比阈值（默认 2.5）
      max_val_rms_px: 容许的验证残差绝对上限（默认 2.0 像素）

    返回：
      dict(passed=bool, cv_ratio=float, rms_train=float, rms_val=float,
           D_train_px=float, verdict=str, status=str)
    """
    radii = np.asarray(ring_radii_px, dtype=float)
    qs = np.asarray(qs, dtype=float)

    m = len(radii)
    if m < 4:
        return dict(
            passed=False,
            cv_ratio=np.nan,
            rms_train=np.nan,
            rms_val=np.nan,
            D_train_px=float(D_px) if D_px is not None else np.nan,
            status="UNDERDETERMINED",
            verdict="环数过少 (m < 4)，无法拆分独立子集验证（欠定简并）"
        )

    # 排序以保证按半径递增
    order = np.argsort(radii)
    radii = radii[order]
    qs = qs[order]

    # 拆分训练集与验证集 (奇偶交替拆分)
    idx_train = np.arange(0, m, 2)
    idx_val = np.arange(1, m, 2)

    r_train, q_train = radii[idx_train], qs[idx_train]
    r_val, q_val = radii[idx_val], qs[idx_val]

    # 理论 tan(2θ) 因子
    tth_train = 2.0 * np.arcsin(np.clip(q_train * lam / (4.0 * np.pi), 0, 0.999))
    k_train = np.tan(tth_train)

    tth_val = 2.0 * np.arcsin(np.clip(q_val * lam / (4.0 * np.pi), 0, 0.999))
    k_val = np.tan(tth_val)

    # 仅使用训练集拟合 D_train (按 1/r² 加权)
    w_train = 1.0 / np.maximum(r_train, 1.0) ** 2
    denom = np.sum(w_train * k_train ** 2)
    if denom <= 0:
        return dict(passed=False, cv_ratio=np.nan, rms_train=np.nan,
                    rms_val=np.nan, status="FAIL", verdict="训练集奇异，无法拟合")

    D_train = float(np.sum(w_train * r_train * k_train) / denom)

    # 训练集内点拟合残差
    res_train = r_train - D_train * k_train
    rms_train = float(np.sqrt(np.mean(res_train ** 2)))

    # 验证集外推残差 (Out-of-sample)
    pred_val = D_train * k_val
    res_val = r_val - pred_val
    rms_val = float(np.sqrt(np.mean(res_val ** 2)))

    # 计算外推残差比 (cv_ratio)
    # 分母加 0.2px 物理地板防止完美拟合时除以零发散
    cv_ratio = float(rms_val / max(rms_train, 0.20))

    passed = (cv_ratio <= max_ratio) and (rms_val <= max_val_rms_px)

    if passed:
        verdict = f"通过: 外推残差比 {cv_ratio:.2f} <= {max_ratio} (验证残差 {rms_val:.2f}px)，未见少环过拟合"
        status = "PASS"
    else:
        verdict = f"警告: 外推残差比 {cv_ratio:.2f} > {max_ratio} 或残差过大 ({rms_val:.2f}px)，疑似落入伪盆地"
        status = "WARN_OVERFIT"

    return dict(
        passed=passed,
        cv_ratio=cv_ratio,
        rms_train=rms_train,
        rms_val=rms_val,
        D_train_px=D_train,
        m_train=len(r_train),
        m_val=len(r_val),
        status=status,
        verdict=verdict
    )
