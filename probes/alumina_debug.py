# -*- coding: utf-8 -*-
"""alumina 精修失败调试：refiner 模型图（GT 几何）vs 数据图的径向剖面对拍。

如果模型环半径 ≠ 数据环半径 → q_map/DetectorBundle 有 bug；
如果对得上 → 问题在 nuisance/优化路径。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio
import torch

from real_probe import load_geom
from torchfai.refine import DifferentiableRefiner
from torchfai.simulator import DetectorBundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AL = os.path.join(ROOT, "data", "real", "alumina")
LAM = 7.084811024e-11
P1, P2 = 103.358e-6, 102.53e-6


def radial_profile(img2d, cy, cx, nb=52, dbin=0.001):
    H, W = img2d.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((yy - cy) * P1) ** 2 + ((xx - cx) * P2) ** 2)
    out = []
    for i in range(nb):
        m = (r >= i * dbin) & (r < (i + 1) * dbin)
        out.append(img2d[m].mean() if m.sum() else np.nan)
    return np.array(out)


def main():
    dark = fabio.open(os.path.join(AL, "dark_0001.edf")).data.astype(np.float64)
    frames = [fabio.open(os.path.join(AL, f"al2o3_{i:04d}.edf")).data
              for i in range(51)]
    img = np.clip(np.mean(frames, 0) - dark, 0, None).astype(np.float32)
    H, W = img.shape
    p = load_geom(os.path.join(AL, "al2o3_00_max_51_frames.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    cy, cx = gt[1] / P1, gt[2] / P2

    yy, xx = np.mgrid[0:H, 0:W]
    db = DetectorBundle(name="Frelon_half", shape=(H, W), pixel=P1,
                        dead=np.zeros((H, W), bool), pyfai_det=None, yy=yy, xx=xx)
    ref = DifferentiableRefiner("Al2O3", os.path.join(ROOT, "data/train_v3"),
                                detector=db, wavelength_m=LAM,
                                dist_range=[0.05, 1.0])

    # 模型图：GT 几何 + 中庸 nuisance
    with torch.no_grad():
        pn = ref.phys_to_pn(gt, ref.consts)
        q = ref.q_map(pn.to(ref.device))
        amps = ref.Is_t * torch.exp(-0.01 * ref.qs_t * ref.qs_t)
        sc = torch.log(torch.tensor(80.0, device=ref.device) / amps.max())
        mu = ref._render(
            q, sc, torch.tensor(0.01, device=ref.device),
            torch.zeros(ref.n_ring, device=ref.device),
            torch.full((ref.n_ring,), float(np.log(0.007)), device=ref.device),
            torch.log(torch.tensor(20.0, device=ref.device)),
            torch.log(torch.tensor(0.05, device=ref.device)),
            torch.log(torch.tensor(1.0, device=ref.device)),
            torch.tensor(1.5, device=ref.device),
            torch.log(torch.tensor(0.25, device=ref.device)),
        ).cpu().numpy()

    prof_d = radial_profile(img, cy, cx)
    prof_m = radial_profile(mu, cy, cx)
    print("r(mm) | 数据 | 模型(GT)   （环半径理论 24.2/33.4/36.0/39.9/41.5/44.3）")
    for i in range(48):
        print(f"{i:5d} | {prof_d[i]:6.1f} | {prof_m[i]:7.1f}"
              + ("   <-- 模型峰" if prof_m[i] == prof_m[max(0,i-2):i+3].max()
                 and prof_m[i] > 90 else ""))
    imax_m = float(np.nanmax(prof_m))
    print(f"\n模型剖面峰位 mm: ",
          [f"{i}" for i in range(48) if prof_m[i] > 0.8 * imax_m][:10])


if __name__ == "__main__":
    main()
