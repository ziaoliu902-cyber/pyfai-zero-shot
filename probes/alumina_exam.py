# -*- coding: utf-8 -*-
"""alumina 第三波长对拍（修复后管线）：Frelon 半CCD @ λ=0.7085Å, dist=117mm 近距。

物理一致输入的特殊处理：Frelon 视域 ~106mm < 训练视域 179mm 且像素 103µm
→ 均匀重采样到 344µm 有效像素（~308×306 网格）→ 零垫到 (521,490)。
（"探测器只覆盖训练世界的中央一块"——训练分布有死区但无整圈边界死区，
 属轻度 OOD，本探针就是来测这个的。）

考卷：51 帧均值 - dark；GT 用 max_51_frames.poni（标定口径一致）。
粗解 + ±8mm 网格精修（refiner 用自建 DetectorBundle，1026×1024，无 spline——
已知局限：spline 未建模，精修残差地板由它决定，报告时注明）。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio
import torch

from real_probe import load_geom
from real_refine_v4 import load_nn
from torchfai.dataset import norm_to_param
from torchfai.refine import DifferentiableRefiner
from torchfai.simulator import DetectorBundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AL = os.path.join(ROOT, "data", "real", "alumina")
LAM = 7.084811024e-11
PX_TR = 344e-6   # 训练有效像素


def main():
    m, dev, consts = load_nn("runs/base_v3lamp/best.pt")

    # ---- 数据：51 帧均值 - dark，中心亮斑按挡光盘语义掩掉 ----
    # （实测：r<6mm 处 >300 计数像素全部集中 = 直束亮斑；训练分布中心是
    #   挡光盘（置零）——不掩则 NN 幻觉远距、精修驼峰吸斑造成 998mm 假谷）
    dark = fabio.open(os.path.join(AL, "dark_0001.edf")).data.astype(np.float64)
    frames = [fabio.open(os.path.join(AL, f"al2o3_{i:04d}.edf")).data
              for i in range(51)]
    img = np.clip(np.mean(frames, 0) - dark, 0, None).astype(np.float32)
    H, W = img.shape
    p1, p2 = 103.358e-6, 102.53e-6
    cy, cx = 0.0529565255372 / p1, 0.05473342483 / p2
    yy0, xx0 = np.mgrid[0:H, 0:W]
    r_m = np.sqrt(((yy0 - cy) * p1) ** 2 + ((xx0 - cx) * p2) ** 2)
    img[r_m < 0.006] = 0.0
    print(f"alumina 图 {img.shape}  均值计数={np.median(img):.2f}  "
          f"峰值={img.max():.0f}  中心亮斑已掩(r<6mm)")

    # ---- NN 输入：均匀重采样到 344µm 有效像素 + 零垫 ----
    from scipy.ndimage import zoom
    p1 = 103.358e-6
    h_t, w_t = int(round(H * p1 / PX_TR)), int(round(W * 102.53e-6 / PX_TR))
    x = zoom(img, (h_t / H, w_t / W), order=1)
    canvas = np.zeros((521, 490), np.float32)
    r0, c0 = (521 - h_t) // 2, (490 - w_t) // 2
    canvas[r0:r0 + h_t, c0:c0 + w_t] = x
    with torch.no_grad():
        t = torch.from_numpy(np.log1p(canvas)).unsqueeze(0).unsqueeze(0).to(dev)
        mu, lv = m(t, torch.tensor([0.70848], device=dev))
    p_nn = norm_to_param(mu.float().cpu().numpy(), consts)[0]

    # ---- GT ----
    p = load_geom(os.path.join(AL, "al2o3_00_max_51_frames.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    print(f"GT: dist={gt[0]*1000:.2f}mm poni=({gt[1]/p1:.0f},{gt[2]/102.53e-6:.0f})px "
          f"rot=({gt[3]:.4f},{gt[4]:.4f})")
    print(f"NN 粗解: dist={p_nn[0]*1000:.1f}mm({(p_nn[0]-gt[0])*1000:+.1f})  "
          f"poni=({p_nn[1]/p1:.0f},{p_nn[2]/102.53e-6:.0f})px")

    # ---- 精修：自建 Frelon 半CCD DetectorBundle（无 spline）----
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    db = DetectorBundle(name="Frelon_half", shape=(H, W), pixel=p1,
                        dead=np.zeros((H, W), bool), pyfai_det=None,
                        yy=yy, xx=xx)
    ref = DifferentiableRefiner("Al2O3", os.path.join(ROOT, "data/train_v3"),
                                detector=db, wavelength_m=LAM,
                                dist_range=[0.05, 1.0])
    best = None
    for dm in (-8.0, -4.0, 0.0, 4.0, 8.0):
        start = np.array([p_nn[0] + dm * 1e-3, H / 2 * p1, W / 2 * 102.53e-6,
                          0.0, 0.0, 0.0])
        r = ref.refine(img, start, iters=400, sigma_norm=None)
        print(f"  起点{dm:+.0f}mm: loss={r['loss']:9.2f} "
              f"distΔ={(r['params'][0]-gt[0])*1000:+8.2f}mm "
              f"poniΔ=({(r['params'][1]-gt[1])/p1:+6.1f},"
              f"{(r['params'][2]-gt[2])/102.53e-6:+6.1f})px "
              f"rot=({r['params'][3]:+.4f},{r['params'][4]:+.4f})", flush=True)
        if best is None or r["loss"] < best["loss"]:
            best = r
    print(f"\n网格最优: distΔ={(best['params'][0]-gt[0])*1000:+.2f}mm "
          f"loss={best['loss']:.2f}（spline 未建模=残差地板，见注）")


if __name__ == "__main__":
    main()
