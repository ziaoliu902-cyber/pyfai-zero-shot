# -*- coding: utf-8 -*-
"""AgBh 真实帧终局协议：直测环心钉 poni + dist 网格精修 + 共识投票。

依据（10-29/30 验尸链）：
  - 自由 poni 精修在少环强背景帧上锁不住束心（+400~700px 乱跑）
  - poni 钉住时 dist 精修稳定收敛（下午 +51.6mm）
  - NN 粗解 v3far3 = 1387mm（GT 1583，-196mm），宽窗 ±700 罩得住
协议：环带质心直测 poni → poni 邻域 ±8px 抖动 × dist 网格 → 每起点
只精修 dist（poni 微调 ±4px 内）→ 投票收敛簇。
"""
import os
import sys

import numpy as np
import torch
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from torchfai.refine import DifferentiableRefiner
from real_refine_v6 import (REAL, load_geom, load_px_nn, match_gain,
                            to_nn_input_phys, nn_coarse)

PX = 172e-6   # Pilatus_1M 像素（本帧原生）


def ring_centroid(img, frac=0.001, rmin=30.0, rmax=220.0, iters=4):
    """高亮环带质心（迭代收敛）：每轮以新质心重画半径带，收敛即止。"""
    H, W = img.shape
    th = np.quantile(img, 1 - frac)
    yy, xx = np.mgrid[0:H, 0:W]
    m = img >= th
    cy, cx = yy[m].mean(), xx[m].mean()
    for _ in range(iters):
        r = np.hypot(yy - cy, xx - cx)
        m2 = m & (r > rmin) & (r < rmax)
        cy, cx = yy[m2].mean(), xx[m2].mean()
    return cy, cx, int(m2.sum())


def main():
    img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
    p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    m, dev, consts = load_px_nn(os.path.join(ROOT, "runs/v3far3/best.pt"))

    # 1. 直测环心
    cy, cx, npx = ring_centroid(img.astype(np.float64))
    poni_gt_px = (gt[1] / PX, gt[2] / PX)
    print(f"[直测环心] ({cy:.0f},{cx:.0f})px  GT ({poni_gt_px[0]:.0f},"
          f"{poni_gt_px[1]:.0f})px  误差=({cy-poni_gt_px[0]:+.0f},"
          f"{cx-poni_gt_px[1]:+.0f})px  样本={npx}", flush=True)

    # 2. NN 粗解 dist（亮度域内 ×1.0）
    p_nn, _ = nn_coarse(m, dev, consts,
                        match_gain(to_nn_input_phys(img)), 1.0, 1.0)
    d0 = p_nn[0]
    print(f"[NN 粗解] dist={d0*1000:.1f}mm（{((d0-gt[0])*1000):+.1f}）",
          flush=True)

    # 3. 精修：poni 直测值 ± 抖动网格 × dist 起点集，只精修 dist
    ref = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3x"),
                                detector="Pilatus_1M",
                                wavelength_m=1.000e-10,
                                dist_range=[0.1, 3.0])
    PIN = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)      # 只放 dist
    res = {}
    for dy, dx in ((0, 0), (-40, 0), (40, 0), (0, -40), (0, 40),
                   (-40, -40), (40, 40), (-40, 40), (40, -40)):
        for dm in (0, 200):
            st = np.array([d0 + dm * 1e-3, (cy + dy) * PX, (cx + dx) * PX,
                           0, 0, 0])
            try:
                out = ref.refine(img, st, iters=200, lr_geo=8e-3,
                                 lr_mult=PIN, sigma_range=(0.0015, 0.6))
                d, l = float(out["params"][0]), float(out["loss"])
                res[(dy, dx, dm)] = (d, l)
                print(f"  poni({dy:+3d},{dx:+3d}) d{dm:+4d} → {d*1000:7.1f}mm "
                      f"({(d-gt[0])*1000:+7.1f})  loss={l:.1f}", flush=True)
            except Exception as e:
                print(f"  poni({dy:+3d},{dx:+3d}) d{dm:+4d} ✗ "
                      f"{type(e).__name__}", flush=True)

    if not res:
        print("全部起点失败")
        return
    # 4. 共识投票：|d - 众数邻域| < 50mm 的簇取中位
    ds = np.array([v[0] for v in res.values()])
    best_k = min(res, key=lambda k: res[k][1])
    best_l = res[best_k]
    med = np.median(ds)
    cl = ds[np.abs(ds - med) < 0.05]
    cons = (float(np.median(cl)) if len(cl) >= 3 else float(best_l[0]),
            len(cl))
    print(f"\n[共识] dist={cons[0]*1000:.1f}mm（{((cons[0]-gt[0])*1000):+.1f}mm）"
          f"  簇={cons[1]}/{len(ds)}")
    print(f"[最优loss] dist={best_l[0]*1000:.1f}mm"
          f"（{((best_l[0]-gt[0])*1000):+.1f}mm）  最优poni抖动=({best_k[0]:+d},"
          f"{best_k[1]:+d})px")
    print(f"GT = {gt[0]*1000:.1f}mm")


if __name__ == "__main__":
    main()
