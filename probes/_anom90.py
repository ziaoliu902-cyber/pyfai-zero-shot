# -*- coding: utf-8 -*-
"""异常复核：D=90mm 相正确率 0%（而 60/130 是 100%）——是相判歧义还是实现问题？

20 seeds @ D=90，逐 seed 打印相/D/中心；对照 D=60、D=130 各 8 seeds。
"""
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from phase_map import run_point, summarize

print("=== D=90mm, 20 seeds ===")
rows = run_point(90, 10000, seeds=20)
byshape = {}
for i, (dD, dc, pok, _) in enumerate(rows):
    print(f"seed{i:02d}: dD={dD:9.2f} dc={dc:7.1f} phase_ok={pok}")
    byshape[pok] = byshape.get(pok, 0) + 1
print("统计:", byshape, "|", {k: (round(float(v), 2) if isinstance(v, float) else v) for k, v in summarize(rows).items() if k != "ok"})

for d in (60, 130):
    print(f"=== D={d}mm, 8 seeds ===")
    r = summarize(run_point(d, 10000, seeds=8))
    print({k: (round(float(v), 2) if isinstance(v, float) else v) for k, v in r.items() if k != "ok"})
