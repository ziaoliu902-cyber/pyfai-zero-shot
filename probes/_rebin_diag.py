# -*- coding: utf-8 -*-
"""诊断：rebin 画布 vs 旧画布 的翻转关系 + 仿射数值抽查。"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI

from real_refine_v6 import to_nn_input_wide

src = open(os.path.join(ROOT, "probes", "_rebin_test.py"),
           encoding="utf-8").read()
code = src.split("print(f\"")[0]
g = {}
exec(compile(code, "rebin_fn", "exec"), g)
rebin_to_canvas = g["rebin_to_canvas"]

img = fabio.open(os.path.join(
    ROOT, "data", "real", "MX_ceria/ceria_350_1_0001.cbf")).data
p = pyFAI.load(os.path.join(
    ROOT, "data", "real", "MX_ceria/ceria_350_1_0001.poni"))
reb = rebin_to_canvas(img, p, 2.425)
wide, k, extra = to_nn_input_wide(img)

print("reb 统计: max", round(float(reb.max()), 0),
      "p99.9", round(float(np.percentile(reb, 99.9)), 0),
      "非零", round(float((reb > 0).mean()), 3))
A = np.log1p(wide)
B = np.log1p(np.clip(reb, 0, None))


def cc(X, Y):
    Xc = X - X.mean()
    Yc = Y - Y.mean()
    return round(float((Xc * Yc).mean() / (Xc.std() * Yc.std() + 1e-9)), 3)


print("恒等:", cc(A, B), " 上下翻:", cc(A, np.flipud(B)),
      " 左右翻:", cc(A, np.fliplr(B)), " 旋转180:", cc(A, np.rot90(B, 2)))

# 仿射抽查：源几个像素的 lab
S = p
pts = [(0, 0), (2526, 0), (0, 2462), (1263, 1231)]
for r, c in pts:
    z, y, x = S.calc_pos_zyx(d1=np.float64(r), d2=np.float64(c))
    print(f"源({r},{c}) lab x={float(np.ravel(x)[0]):+.4f} "
          f"y={float(np.ravel(y)[0]):+.4f} z={float(np.ravel(z)[0]):+.4f}")
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from torchfai.simulator import get_scaled_detector
db = get_scaled_detector("Pilatus_1M", 2, 2.425)
pdet = Detector(pixel1=db.pixel, pixel2=db.pixel, max_shape=tuple(db.shape))
D = AzimuthalIntegrator(dist=S.dist, poni1=S.poni1, poni2=S.poni2,
                        rot1=S.rot1, rot2=S.rot2, rot3=S.rot3,
                        wavelength=S.wavelength, detector=pdet)
print("dest orientation:", D.detector.orientation)
for r, c in [(0, 0), (520, 0), (0, 489), (260, 245)]:
    z, y, x = D.calc_pos_zyx(d1=np.float64(r), d2=np.float64(c))
    print(f"目({r},{c}) lab x={float(np.ravel(x)[0]):+.4f} "
          f"y={float(np.ravel(y)[0]):+.4f} z={float(np.ravel(z)[0]):+.4f}")
