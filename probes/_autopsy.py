# -*- coding: utf-8 -*-
"""AgBh/400mm 验尸：饱和占比 + poni 钉 GT 对照 + include_sat 消融。"""
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

# ---- AgBh ----
ref2 = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus_1M", wavelength_m=1.000e-10,
                             dist_range=[0.1, 3.0])
img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data.astype(np.float64)
p = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))
gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
sat = float(ref2.cfg["sat_clip"])
print(f"AgBh: sat_clip={sat}  帧max={img.max():.0f}  "
      f"饱和占比={(img >= sat).mean():.4f}  >10万占比={(img > 1e5).mean():.5f}")
H2, W2 = ref2.db.extent
p_nn, _ = nn_coarse(m, dev, consts,
                    match_gain(to_nn_input_phys(img)) * 0.5, 1.0, 1.0)
print(f"NN 粗解 dist={p_nn[0]*1000:.1f} (GT {gt[0]*1000:.1f}) "
      f"poni 误差=({(p_nn[1]-gt[1])/172e-6:+.1f},{(p_nn[2]-gt[2])/172e-6:+.1f})px")
for tag, kw in (("标准", {}),
                ("include_sat", dict(include_sat=True))):
    st = np.array([p_nn[0], gt[1], gt[2], 0, 0, 0])   # poni 钉 GT：诊断用
    r = ref2.refine(img, st, iters=400, sigma_norm=None,
                    sigma_range=(0.0015, 0.02), **kw)
    print(f"  [{tag}] poni@GT: distΔ={(r['params'][0]-gt[0])*1000:+8.2f}mm "
          f"loss={r['loss']:.1f} "
          f"终点poni=({(r['params'][1]-gt[1])/172e-6:+.0f},"
          f"{(r['params'][2]-gt[2])/172e-6:+.0f})px")

# ---- ceria 400mm ----
ref6 = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_px1"),
                             detector="Pilatus6M", wavelength_m=9.72386e-11,
                             dist_range=[0.1, 1.0])
img = fabio.open(os.path.join(REAL, "MX_ceria/ceria_400_1_0001.cbf")).data
p = pyFAI.load(os.path.join(REAL, "MX_ceria/ceria_400_1_0001.poni"))
gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
print(f"\nceria400: 帧max={img.max()} sat_clip={float(ref6.cfg['sat_clip'])} "
      f"饱和占比={(img >= float(ref6.cfg['sat_clip'])).mean():.5f}")
for tag, st_poni in (("poni@GT", (gt[1], gt[2])),
                     ("poni@GT+10px", (gt[1] + 10 * 172e-6, gt[2] + 10 * 172e-6)),
                     ("poni@GT-10px", (gt[1] - 10 * 172e-6, gt[2] - 10 * 172e-6))):
    st = np.array([0.40, st_poni[0], st_poni[1], 0, 0, 0])
    r = ref6.refine(img, st, iters=400, sigma_norm=None,
                    sigma_range=(0.0015, 0.02))
    print(f"  [{tag}] distΔ={(r['params'][0]-gt[0])*1000:+8.2f}mm "
          f"loss={r['loss']:.2f} "
          f"终点poni=({(r['params'][1]-gt[1])/172e-6:+.0f},"
          f"{(r['params'][2]-gt[2])/172e-6:+.0f})px")
