# -*- coding: utf-8 -*-
"""_tmp_armB_canvas.py：B 臂稳健性——to_canvas 朴素池化画布（alumina 生产同款）。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
import fabio
import pyFAI

from solver_core import solve, ring_radii, to_canvas
from torchfai.phases import phase_qs

LAM = 0.972386
Q = phase_qs("CeO2")[:8]
print("B臂稳健性: to_canvas(朴素池化) + 几何中心 + 全域投票")
es = []
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    canvas, s, cx0, cy0 = to_canvas(img)
    pxm = 172e-6 * s
    pk = ring_radii(canvas, cx0, cy0)[0]
    sol = solve(pk, cx0, cy0, 1.0 / pxm, Q, LAM, pxm,
                d_range=(0.05, 3.0), rms_floor=1.8)
    if sol is None:
        print(f"{d}: 失败")
        es.append(999.0)
        continue
    e = sol[2] * pxm * 1000 - gt
    es.append(abs(e))
    print(f"{d}: D={sol[2]*pxm*1000:.1f}mm Δ={e:+.1f} m={sol[3]} "
          f"rms={sol[4]:.2f}px")
ok = np.mean(es) < 10 and max(es) < 15
print(f"聚合: 平均{np.mean(es):.1f} 最差{max(es):.1f} → "
      f"{'✓过' if ok else '✗未过'}")
