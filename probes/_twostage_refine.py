# -*- coding: utf-8 -*-
"""两段式精修终局测试：dist 独走（poni/rots 冻结）→ 全参数落座。

机理（10-29 下午定位）：六维峡谷中 poni 鞍点陷阱——起点 dist 偏低时，
优化器用 poni 妥协换取短程降损，卡进 dist-假盆地（200/250/400 卡
-29/-46/-48mm，2000 迭代不收敛=盆地非迭代问题）。先锁 poni 让 dist
独立走完，再放开全参数细磨。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
import fabio
import pyFAI

from real_refine_v5 import to_nn_input_phys
from real_refine_v6 import (REAL, ROOT, load_px_nn, match_gain, nn_coarse,
                            to_nn_input_wide)
from torchfai.refine import DifferentiableRefiner

m, dev, consts = load_px_nn(os.path.join(ROOT, "runs/px_v2/best.pt"))
ref6 = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus6M", wavelength_m=9.72386e-11,
                             dist_range=[0.1, 1.0])

FREEZE = (1.0, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5)
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    wide, k, extra = to_nn_input_wide(img)
    p_nn, _ = nn_coarse(m, dev, consts, wide, 0.972386, k)
    st = np.array([p_nn[0], gt[1], gt[2], 0.0, 0.0, 0.0])   # NN dist + 标称中心
    r1 = ref6.refine(img, st, iters=250, lr_mult=FREEZE, sigma_norm=None,
                     sigma_range=(0.0015, 0.02))
    r2 = ref6.refine(img, r1["params"], iters=400, sigma_norm=None,
                     sigma_range=(0.0015, 0.02))
    px = 172e-6
    print(f"[ceriaW {d}mm 两段式] 段1 distΔ={(r1['params'][0]-gt[0])*1000:+.2f}"
          f" → 段2 {(r2['params'][0]-gt[0])*1000:+.2f}mm "
          f"poni=({(r2['params'][1]-gt[1])/px:+.1f},"
          f"{(r2['params'][2]-gt[2])/px:+.1f})px loss={r2['loss']:.2f}",
          flush=True)

print("-- AgBh@1M 两段式 --")
ref2 = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus_1M", wavelength_m=1.000e-10,
                             dist_range=[0.1, 3.0])
img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
p = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))
gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
H2, W2 = ref2.db.extent
p_nn, _ = nn_coarse(m, dev, consts,
                    match_gain(to_nn_input_phys(img)) * 0.5, 1.0, 1.0)
st = np.array([p_nn[0], H2 / 2, W2 / 2, 0.0, 0.0, 0.0])
r1 = ref2.refine(img, st, iters=250, lr_mult=FREEZE, sigma_norm=None,
                 sigma_range=(0.0015, 0.02))
r2 = ref2.refine(img, r1["params"], iters=400, sigma_norm=None,
                 sigma_range=(0.0015, 0.02))
print(f"[AgBh 两段式] 段1 {(r1['params'][0]-gt[0])*1000:+.2f}"
      f" → 段2 {(r2['params'][0]-gt[0])*1000:+.2f}mm loss={r2['loss']:.2f}",
      flush=True)
