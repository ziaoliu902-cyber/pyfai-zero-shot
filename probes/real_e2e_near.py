# -*- coding: utf-8 -*-
"""端到端验证：物理一致输入的 NN 粗解 → 纯物理精修，ceria 近距真实图。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio
import pyFAI
import torch

from real_crop_probe import bin2
from real_refine_v4 import load_nn
from torchfai.refine import DifferentiableRefiner
from torchfai.dataset import norm_to_param

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")
LAM_CERIA = 9.72386e-11


def main():
    m, dev, consts = load_nn("runs/base_v3lamp/best.pt")
    ref = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3"),
                                detector="Pilatus6M", wavelength_m=LAM_CERIA,
                                dist_range=[0.1, 1.0])
    for d in (150, 200, 250, 300):
        img = fabio.open(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        H, W = img.shape
        r0, c0 = H // 2 - 521, W // 2 - 490
        x = bin2(img[r0:r0 + 1042, c0:c0 + 980])
        with torch.no_grad():
            t = torch.from_numpy(np.log1p(x)).unsqueeze(0).unsqueeze(0).to(dev)
            mu, _ = m(t, torch.tensor([0.972386], device=dev))
        p_nn = norm_to_param(mu.float().cpu().numpy(), consts)[0]
        # poni 物理量出域（NN 域=训练视域），精修起点用探测器中心 + NN dist
        Hm, Wm = ref.db.extent
        best = None
        for dm in (-8.0, -4.0, 0.0, 4.0, 8.0):   # 密网格 mm 偏移（盆半宽~4mm）
            start = np.array([p_nn[0] + dm * 1e-3, Hm / 2, Wm / 2, 0.0, 0.0, 0.0])
            r = ref.refine(img, start, iters=400, sigma_norm=None)
            if best is None or r["loss"] < best[2]["loss"]:
                best = (dm, start, r)
        f, start, r = best
        px = ref.db.pixel
        print(f"[ceria {d}mm] NN粗解={p_nn[0]*1000:.1f}({(p_nn[0]-gt[0])*1000:+.1f}) "
              f"{f:+.0f}mm → 精修后 distΔ={(r['params'][0]-gt[0])*1000:+.2f}mm "
              f"poniΔ=({(r['params'][1]-gt[1])/px:+.1f},{(r['params'][2]-gt[2])/px:+.1f})px "
              f"loss={r['loss']:.2f}", flush=True)


if __name__ == "__main__":
    main()
