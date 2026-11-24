# -*- coding: utf-8 -*-
"""shared_tilt.py：变距扫描 / 导轨扫描下的倾角自由度共享联合标定模块

物理背景：
  在同步辐射与实验室 X 射线实验中，经常需要在不同样品-探测器距离下测量标样
  （变距扫描，Distance Scan）。
  探测器沿光轨移动时：
  1. 探测器倾角 (rot1, rot2) 由安装机械结构决定，在全行程中保持恒定（共享自由度）；
  2. 距离 D_k 逐帧改变（各帧独立未知数）；
  3. 束心 (p1_k, p2_k) 可能因导轨与光轴不严格平行而存在微小线性走位（可配共享或每帧独立）。

本模块提供：
  - joint_solve_multidist_tilt(): 变距共享倾角拟合
  - joint_solve_multidist_tilt_robust(): 抗差版（自动剔除局部污染/错解帧）
"""

import sys
import os
import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def joint_solve_multidist_tilt(cpts_list, px1, px2, shared_center=False,
                               boot=6, seed=3, spline=None):
    """变距扫描下的多帧联合标定：全局共享 (rot1, rot2) + 各帧独立距离 (D_k) + 中心 (p1_k, p2_k)。

    参数：
      cpts_list: 各帧控制点 dict 列表（由 cpts_of_frame 生成）。
                 每帧需包含 d1, d2, tth, w, p1, p2, D(mm)。
      px1, px2: 像素物理尺寸（米）
      shared_center: 是否强制全距离共用同一束心（若光轨机械精度极高可选 True）
      boot: bootstrap 抽样轮数（>=1 时评估各量不确定度）
      seed: 随机种子
      spline: 探测器畸变文件路径

    返回：
      dict 包含全局 rot1, rot2, 逐帧 dists(米), dists_mm(毫米), p1s, p2s, rms 等，
      失败返回 None。
    """
    from poni_out import make_pyfai_fwd

    live = [c for c in cpts_list if c is not None and len(c.get("d1", ())) >= 10]
    K = len(live)
    if K < 2:
        return None

    fwd = make_pyfai_fwd(px1, px2, spline)

    D1 = np.concatenate([c["d1"] for c in live])
    D2 = np.concatenate([c["d2"] for c in live])
    TT = np.concatenate([c["tth"] for c in live])
    WW = np.concatenate([c["w"] for c in live])
    WW = WW / max(WW.mean(), 1e-12)

    FR = np.concatenate([[k] * len(c["d1"]) for k, c in enumerate(live)])

    # 初值构建
    # rot1=0.0, rot2=0.0
    r1_0, r2_0 = 0.0, 0.0
    dists0 = [float(c.get("D", 200.0)) * 1e-3 for c in live]  # 转换为米

    if not shared_center:
        # 参数结构: [rot1, rot2, D_0, p1_0, p2_0, D_1, p1_1, p2_1, ...]
        p0 = [r1_0, r2_0]
        lo = [-0.10, -0.10]
        hi = [0.10, 0.10]
        for c, d0 in zip(live, dists0):
            p1_0 = float(c.get("p1", 0.1))
            p2_0 = float(c.get("p2", 0.1))
            p0.extend([d0, p1_0, p2_0])
            lo.extend([max(0.01, d0 * 0.5), p1_0 - 0.05, p2_0 - 0.05])
            hi.extend([d0 * 2.0, p1_0 + 0.05, p2_0 + 0.05])

        p0 = np.asarray(p0, dtype=np.float64)
        bounds = (np.asarray(lo, float), np.asarray(hi, float))

        def resid(p):
            r1, r2 = p[0], p[1]
            # 取各点所属帧对应的 D, p1, p2
            D_vec = p[2 + 3 * FR]
            p1_vec = p[3 + 3 * FR]
            p2_vec = p[4 + 3 * FR]
            tth_pred = fwd(D_vec, p1_vec, p2_vec, r1, r2, D1, D2)
            return (tth_pred - TT) * WW

    else:
        # 共享中心参数结构: [rot1, rot2, p1, p2, D_0, D_1, ...]
        p1_mean = float(np.mean([c.get("p1", 0.1) for c in live]))
        p2_mean = float(np.mean([c.get("p2", 0.1) for c in live]))
        p0 = [r1_0, r2_0, p1_mean, p2_mean] + dists0
        lo = [-0.10, -0.10, p1_mean - 0.05, p2_mean - 0.05] + [max(0.01, d * 0.5) for d in dists0]
        hi = [0.10, 0.10, p1_mean + 0.05, p2_mean + 0.05] + [d * 2.0 for d in dists0]

        p0 = np.asarray(p0, dtype=np.float64)
        bounds = (np.asarray(lo, float), np.asarray(hi, float))

        def resid(p):
            r1, r2 = p[0], p[1]
            p1, p2 = p[2], p[3]
            D_vec = p[4 + FR]
            tth_pred = fwd(D_vec, p1, p2, r1, r2, D1, D2)
            return (tth_pred - TT) * WW

    try:
        res = least_squares(resid, p0, bounds=bounds, method="trf",
                            loss="soft_l1", f_scale=1e-4, max_nfev=3000)
    except Exception:
        return None

    if not res.success:
        return None

    p_opt = res.x
    rot1_opt = float(p_opt[0])
    rot2_opt = float(p_opt[1])

    if not shared_center:
        dists_opt = [float(p_opt[2 + 3 * k]) for k in range(K)]
        p1s_opt = [float(p_opt[3 + 3 * k]) for k in range(K)]
        p2s_opt = [float(p_opt[4 + 3 * k]) for k in range(K)]
    else:
        dists_opt = [float(p_opt[4 + k]) for k in range(K)]
        p1s_opt = [float(p_opt[2])] * K
        p2s_opt = [float(p_opt[3])] * K

    rms_rad = float(np.sqrt(np.mean((res.fun / WW) ** 2)))

    out = dict(
        rot1=rot1_opt,
        rot2=rot2_opt,
        rot1_deg=float(np.rad2deg(rot1_opt)),
        rot2_deg=float(np.rad2deg(rot2_opt)),
        dists=dists_opt,
        dists_mm=[d * 1e3 for d in dists_opt],
        p1s=p1s_opt,
        p2s=p2s_opt,
        rms=rms_rad,
        npts=int(len(TT)),
        nf=K,
        shared_center=shared_center,
        mode="shared_tilt"
    )

    # Bootstrap 评估不确定度
    if boot >= 2:
        rng = np.random.default_rng(seed)
        boot_rot1 = []
        boot_rot2 = []
        boot_dists = [[] for _ in range(K)]

        for _ in range(boot):
            # 每帧按点采样 75%
            sel_indices = []
            for k in range(K):
                k_idx = np.nonzero(FR == k)[0]
                if len(k_idx) > 10:
                    sub = rng.choice(k_idx, size=max(8, int(len(k_idx) * 0.75)), replace=False)
                    sel_indices.extend(sub)
            sel_indices = np.asarray(sel_indices, int)
            if len(sel_indices) < 30:
                continue

            FR_b = FR[sel_indices]
            D1_b = D1[sel_indices]
            D2_b = D2[sel_indices]
            TT_b = TT[sel_indices]
            WW_b = WW[sel_indices]

            if not shared_center:
                def rb(pp):
                    return (fwd(pp[2 + 3 * FR_b], pp[3 + 3 * FR_b], pp[4 + 3 * FR_b],
                                pp[0], pp[1], D1_b, D2_b) - TT_b) * WW_b
            else:
                def rb(pp):
                    return (fwd(pp[4 + FR_b], pp[2], pp[3],
                                pp[0], pp[1], D1_b, D2_b) - TT_b) * WW_b

            try:
                fit_b = least_squares(rb, p_opt, bounds=bounds, method="trf",
                                      loss="soft_l1", max_nfev=500)
                if fit_b.success:
                    boot_rot1.append(float(fit_b.x[0]))
                    boot_rot2.append(float(fit_b.x[1]))
                    if not shared_center:
                        for k in range(K):
                            boot_dists[k].append(float(fit_b.x[2 + 3 * k]) * 1e3)
                    else:
                        for k in range(K):
                            boot_dists[k].append(float(fit_b.x[4 + k]) * 1e3)
            except Exception:
                pass

        if len(boot_rot1) >= 2:
            out["sigma_rot1"] = float(np.std(boot_rot1))
            out["sigma_rot2"] = float(np.std(boot_rot2))
            out["sigmas_D_mm"] = [float(np.std(bd)) if len(bd) >= 2 else 0.0 for bd in boot_dists]

    return out


def joint_solve_multidist_tilt_robust(cpts_list, px1, px2, shared_center=False,
                                      boot=6, seed=3, spline=None,
                                      k_sigma=3.0, max_iter=2, min_frames=2):
    """抗差版变距共享倾角拟合：迭代剔除离群/失锁帧。"""
    from poni_out import make_pyfai_fwd

    live_idx = [i for i, c in enumerate(cpts_list) if c is not None and len(c.get("d1", ())) >= 10]
    if len(live_idx) < min_frames:
        return None

    out = joint_solve_multidist_tilt(cpts_list, px1, px2, shared_center=shared_center,
                                     boot=boot, seed=seed, spline=spline)
    if out is None:
        return None

    fwd = make_pyfai_fwd(px1, px2, spline)
    dropped = []

    for _ in range(max_iter):
        live = [cpts_list[i] for i in live_idx]
        D1 = np.concatenate([c["d1"] for c in live])
        D2 = np.concatenate([c["d2"] for c in live])
        TT = np.concatenate([c["tth"] for c in live])
        W = np.concatenate([c["w"] for c in live])
        W = W / max(W.mean(), 1e-12)
        FR = np.concatenate([[k] * len(c["d1"]) for k, c in enumerate(live)])

        rot1 = out["rot1"]
        rot2 = out["rot2"]
        dists = np.asarray(out["dists"], float)
        p1s = np.asarray(out["p1s"], float)
        p2s = np.asarray(out["p2s"], float)

        if len(dists) != len(live):
            break

        pred = fwd(dists[FR], p1s[FR], p2s[FR], rot1, rot2, D1, D2)
        resid = (pred - TT) * W

        # 逐帧评估均方根残差
        frame_rms = []
        for k in range(len(live)):
            mask = (FR == k)
            if mask.sum() > 0:
                frame_rms.append(float(np.sqrt(np.mean(resid[mask] ** 2))))
            else:
                frame_rms.append(np.inf)
        frame_rms = np.asarray(frame_rms, float)

        valid_rms = frame_rms[np.isfinite(frame_rms)]
        if len(valid_rms) == 0:
            break
        med = float(np.median(valid_rms))
        if med <= 0:
            break

        keep = np.isfinite(frame_rms) & (frame_rms <= k_sigma * med)
        if keep.all() or keep.sum() < min_frames:
            break

        # 记录被剔除的帧索引
        drop_idx = [live_idx[k] for k in np.nonzero(~keep)[0]]
        dropped.extend(drop_idx)
        live_idx = [live_idx[k] for k in np.nonzero(keep)[0]]

        sub_list = [cpts_list[i] if i in set(live_idx) else None for i in range(len(cpts_list))]
        out = joint_solve_multidist_tilt(sub_list, px1, px2, shared_center=shared_center,
                                         boot=boot, seed=seed, spline=spline)
        if out is None:
            break

    if out is not None:
        out = dict(out)
        out["dropped"] = dropped
        out["nf_kept"] = len(live_idx)

    return out
