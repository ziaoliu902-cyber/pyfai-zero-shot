# -*- coding: utf-8 -*-
"""AgBh 增益 × (dist, poni) 敏感性扫描：×0.5 fudge 是否弄瞎了 poni。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
import fabio
import pyFAI

from real_refine_v5 import to_nn_input_phys
from real_refine_v6 import REAL, ROOT, load_px_nn, match_gain, nn_coarse

m, dev, consts = load_px_nn(os.path.join(ROOT, "runs/px_v2/best.pt"))
img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
p = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))
gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])

raw = to_nn_input_phys(img)
g = match_gain(raw)                     # = raw × (20536/p99.9) ≈ raw×0.69
print(f"raw p99.9={np.percentile(raw, 99.9):.0f}  match_gain 净倍率≈0.69")
print(f"{'档位':<14}{'p99.9':>8}{'distΔ':>9}{'poniΔ_y':>9}{'poniΔ_x':>9}")
for nm, v in (("raw×0.25", raw * 0.25), ("raw×0.5", raw * 0.5),
              ("raw×0.7", raw * 0.7), ("raw×1.0", raw),
              ("raw×1.4", raw * 1.4), ("raw×2.0", raw * 2.0),
              ("g×0.5(fudge)", g * 0.5), ("g×1.0", g)):
    p_nn, _ = nn_coarse(m, dev, consts, v, 1.0, 1.0)
    print(f"{nm:<14}{np.percentile(v, 99.9):>8.0f}"
          f"{(p_nn[0]-gt[0])*1000:>+9.1f}"
          f"{(p_nn[1]-gt[1])/172e-6:>+9.1f}"
          f"{(p_nn[2]-gt[2])/172e-6:>+9.1f}")
