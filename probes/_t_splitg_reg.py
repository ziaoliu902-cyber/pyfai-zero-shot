# -*- coding: utf-8 -*-
"""_t_splitg_reg.py:_SPLIT_G=True 默认后的基线回归(2026-11-07)。

B12① 判决开 splitG;回归确认对称域(ceria/P2M)无损:
  P2M CdTe CeO2 GT 800mm(λ/E e2e 基线 799.888)→ 应仍在 ±0.15mm
  ceria_0000(GT?poni 读)抽查一帧。
"""
import glob
import os
import re
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc

assert sc._SPLIT_G is True, "默认未开!"
print(f"_SPLIT_G={sc._SPLIT_G}(新默认)回归:")

# P2M
img = fabio.open(ROOT + r"\data\real\Pilatus2MCdTe_ID15_CeO2_72100eV_"
                 r"800mm_0000.cbf").data.astype(np.float64)
img = np.where(np.isfinite(img), img, 0.0)
r = sc.solve_image(img, 172e-6, lam=12.39842 / 72.1)
D = r["sol"][2] * r["s"] * 172e-3
print(f"  P2M GT800: D={D:.3f}mm Δ={D-800:+.3f}(基线 +0.112) m={r['m']}")
ok1 = abs(D - 800) < 0.15

# ceria 0000(读 GT)
def poni_dist(p):
    for ln in open(p, encoding="utf-8", errors="replace"):
        if ln.lower().startswith("distance"):
            return float(ln.split(":", 1)[1]) * 1000
    return None

for f in sorted(glob.glob(ROOT + r"\data\real\MX_ceria\ceria_350_1_0001.cbf")):
    gt = poni_dist(f.replace(".cbf", ".poni"))
    img = fabio.open(f).data.astype(np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    r = sc.solve_image(img, 172e-6, lam=0.972386)
    D = r["sol"][2] * r["s"] * 172e-3
    print(f"  ceria GT{gt:.0f}: D={D:.3f}mm Δ={D-gt:+.3f} m={r['m']}")
    ok2 = abs(D - gt) < 0.5

print("\n回归:" + ("✓ 通过" if ok1 and ok2 else "✗ 破基线,需回退"))
sys.exit(0 if ok1 and ok2 else 1)
