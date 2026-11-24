# -*- coding: utf-8 -*-
"""azimuthal_stacking.py：多帧方位角堆叠与部分残弧（Partial-Arc）拼接重构模块

理论背景：
  当束心位于探测器边缘或四角时（如 DESY SDD350 corner、Fairchild 等），衍射环在单帧上
  仅呈现部分残弧（方位角覆盖率 < 40%）。
  在单帧上，束心位移与样品距离 D 沿残弧法线方向存在精确的相互代偿，导致单一解在数学上
  欠定/不可辨识（产生高度自洽但实际偏差可达 25% 的伪解）。

解决方案（多帧方位角堆叠）：
  通过多帧测量（如样品台方位旋转 φ、探测器平移移动、或不同曝光角度）：
  1. measure_azimuthal_coverage(): 量化各帧残弧的方位角覆盖率与开角；
  2. stack_azimuthal_frames(): 将多帧互补的残弧进行极坐标对齐与多重曝光融合，合成“虚拟全环图”；
  3. joint_solve_partial_arcs(): 依托合成全环或互补法向约束，彻底打破 D 与束心的简并代偿，
     高精度恢复真实几何距离。
"""

import os
import sys
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def measure_azimuthal_coverage(img, cx, cy, n_sectors=36, r_min=30, r_max=None,
                               threshold_factor=0.15):
    """测量图像在给定中心 (cx, cy) 下的衍射环方位角覆盖率。

    参数：
      img: 2D numpy 图像
      cx, cy: 像素坐标束心（x=列, y=行）
      n_sectors: 方位角划分扇区数（默认 36 槽，每槽 10°）
      r_min: 最小半径（避开直射束饱和区）
      r_max: 最大半径（None 为至探测器边缘）
      threshold_factor: 环强度判定阈值系数

    返回：
      dict(coverage=0~1, is_partial_arc=bool, active_sectors=int,
           total_sectors=int, arc_span_deg=float, sector_strengths=ndarray)
    """
    H, W = img.shape
    if r_max is None:
        r_max = float(np.hypot(max(cx, W - cx), max(cy, H - cy)))

    y, x = np.mgrid[0:H, 0:W]
    r = np.hypot(x - cx, y - cy)
    phi = np.arctan2(y - cy, x - cx)  # [-pi, pi]

    valid = (r >= r_min) & (r <= r_max) & np.isfinite(img) & (img > 0)
    if valid.sum() < 100:
        return dict(
            coverage=0.0,
            is_partial_arc=True,
            active_sectors=0,
            total_sectors=n_sectors,
            arc_span_deg=0.0,
            sector_strengths=np.zeros(n_sectors)
        )

    # 划分扇区
    phi_norm = (phi[valid] + np.pi) / (2 * np.pi)  # [0, 1)
    sec_idx = np.clip((phi_norm * n_sectors).astype(int), 0, n_sectors - 1)

    vals = img[valid]
    # 计算每个扇区的特征强度（取 90 分位数减去中位数背景）
    strengths = np.zeros(n_sectors, dtype=float)
    counts = np.bincount(sec_idx, minlength=n_sectors)

    for s in range(n_sectors):
        m = (sec_idx == s)
        if m.sum() >= 20:
            v_sec = vals[m]
            bg = float(np.median(v_sec))
            # 衍射环在扇区内通常只占 0.5%~1% 的面积，故采用 99.5% 高分位数提取环峰
            hi = float(np.percentile(v_sec, 99.5))
            strengths[s] = max(0.0, hi - bg)

    max_str = float(np.max(strengths))
    if max_str <= 0:
        active = np.zeros(n_sectors, dtype=bool)
    else:
        # 大于最大扇区强度的 threshold_factor 判为包含有效衍射环
        active = (strengths >= threshold_factor * max_str)

    active_count = int(np.sum(active))
    coverage = float(active_count / n_sectors)

    # 计算连续最大弧度跨度（考虑循环）
    if active_count == 0:
        max_consec = 0
    else:
        doubled = np.concatenate([active, active])
        runs = []
        cur = 0
        for b in doubled:
            if b:
                cur += 1
                runs.append(cur)
            else:
                cur = 0
        max_consec = min(n_sectors, max(runs) if runs else 0)

    arc_span_deg = float(max_consec * (360.0 / n_sectors))
    is_partial = (coverage < 0.40) or (arc_span_deg < 150.0)

    return dict(
        coverage=coverage,
        is_partial_arc=is_partial,
        active_sectors=active_count,
        total_sectors=n_sectors,
        arc_span_deg=arc_span_deg,
        sector_strengths=strengths
    )


def polar_transform(img, cx, cy, r_min=10, r_max=None, nr=512, nphi=720):
    """将直角坐标图像投影至极坐标矩阵 (nr, nphi)。

    行表示半径 r，列表示方位角 phi (0 -> 2*pi)。
    """
    H, W = img.shape
    if r_max is None:
        r_max = float(np.hypot(max(cx, W - cx), max(cy, H - cy)))

    rs = np.linspace(r_min, r_max, nr)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    R, PHI = np.meshgrid(rs, phis, indexing="ij")

    # 映射回图像像素 (x, y)
    X = cx + R * np.cos(PHI)
    Y = cy + R * np.sin(PHI)

    polar = map_coordinates(img, [Y, X], order=1, mode="constant", cval=0.0)
    return polar, rs, phis


def stack_azimuthal_frames(imgs, cxs=None, cys=None, angles_deg=None,
                           shifts=None, method="max"):
    """多帧方位角堆叠融合。

    支持两种典型实验场景：
      1. 样品/探测器方位角旋转（如 4 帧各转 90°）：提供 angles_deg=[0, 90, 180, 270]；
      2. 探测器平移拼接（如束心在不同角）：提供 cxs, cys 或 shifts=[(dx, dy), ...]。

    参数：
      imgs: 图像列表（numpy 数组）
      cxs, cys: 每帧对应的束心坐标列表（像素）
      angles_deg: 每帧的相对旋转角（度）
      shifts: 每帧相对于基准帧的平移 (dx, dy)
      method: "max" (极大值融合) 或 "mean" (加权平均)

    返回：
      composite_img: 堆叠合成后的 2D 衍射图
      report: 堆叠前后方位角覆盖率对比及质检信息
    """
    K = len(imgs)
    if K == 0:
        raise ValueError("imgs 列表不能为空")
    if K == 1:
        return imgs[0], {"coverage_before": [1.0], "coverage_after": 1.0}

    H, W = imgs[0].shape
    ref_cx = cxs[0] if cxs is not None else W / 2.0
    ref_cy = cys[0] if cys is not None else H / 2.0

    # 测量各帧原始方位角覆盖率
    cov_before = []
    for k in range(K):
        cx_k = cxs[k] if cxs is not None else ref_cx
        cy_k = cys[k] if cys is not None else ref_cy
        c_k = measure_azimuthal_coverage(imgs[k], cx_k, cy_k)
        cov_before.append(c_k["coverage"])

    # 统一画幅与累积网格
    # 若存在旋转角度，对每帧执行逆旋转对齐到基准坐标系
    transformed_frames = []
    y_grid, x_grid = np.mgrid[0:H, 0:W]

    for k in range(K):
        img_k = np.asarray(imgs[k], dtype=np.float32)
        cx_k = cxs[k] if cxs is not None else ref_cx
        cy_k = cys[k] if cys is not None else ref_cy

        ang_deg = angles_deg[k] if angles_deg is not None else 0.0
        ang_rad = np.deg2rad(ang_deg)

        dx = shifts[k][0] if shifts is not None else (cx_k - ref_cx)
        dy = shifts[k][1] if shifts is not None else (cy_k - ref_cy)

        if abs(ang_rad) < 1e-4 and abs(dx) < 1e-3 and abs(dy) < 1e-3:
            transformed_frames.append(img_k)
            continue

        # 逆坐标变换矩阵
        # (x - ref_cx, y - ref_cy) -> 旋转 + 平移 -> (x_src, y_src)
        cos_a = np.cos(ang_rad)
        sin_a = np.sin(ang_rad)

        rel_x = x_grid - ref_cx
        rel_y = y_grid - ref_cy

        # 逆旋转
        src_rel_x = rel_x * cos_a + rel_y * sin_a
        src_rel_y = -rel_x * sin_a + rel_y * cos_a

        src_x = src_rel_x + cx_k
        src_y = src_rel_y + cy_k

        aligned = map_coordinates(img_k, [src_y, src_x], order=1, mode="constant", cval=0.0)
        transformed_frames.append(aligned)

    stacked_stack = np.stack(transformed_frames, axis=0)

    if method == "max":
        composite = np.max(stacked_stack, axis=0)
    else:
        # 有效像素计数平均
        mask_valid = (stacked_stack > 0)
        cnt = np.sum(mask_valid, axis=0)
        sum_v = np.sum(stacked_stack, axis=0)
        composite = np.zeros_like(sum_v)
        nz = cnt > 0
        composite[nz] = sum_v[nz] / cnt[nz]

    # 测量堆叠后的方位角覆盖率
    cov_after_dict = measure_azimuthal_coverage(composite, ref_cx, ref_cy)
    cov_after = cov_after_dict["coverage"]

    report = dict(
        nf=K,
        coverage_before=cov_before,
        coverage_after=cov_after,
        arc_span_after_deg=cov_after_dict["arc_span_deg"],
        is_partial_resolved=(not cov_after_dict["is_partial_arc"]),
        method=method
    )

    return composite, report


def solve_stacked_partial_arcs(frames, px_m, lam, phase_name, angles_deg=None,
                               shifts=None, cxs=None, cys=None):
    """端到端残弧多帧堆叠求解函数：

    将多帧残弧融合成全环后调用主求解器 solve_image，打破欠定简并。
    """
    from solver_core import solve_image

    composite, report = stack_azimuthal_frames(
        frames, cxs=cxs, cys=cys, angles_deg=angles_deg, shifts=shifts, method="max"
    )

    sol_res = solve_image(composite, px_m=px_m, lam=lam, want=phase_name)
    sol_res["azimuthal_stack_report"] = report

    return sol_res
