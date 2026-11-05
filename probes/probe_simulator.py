# -*- coding: utf-8 -*-
"""simulator.py 冒烟测试：固定种子生成小批量，检查输出结构与统计合理性。跑完即弃。"""
import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Win GBK 控制台兼容

from torchfai.config import SimConfig
from torchfai.simulator import get_detector, synth_sample

cfg = SimConfig()
db = get_detector(cfg.detector_name, cfg.binning)
print(f"detector {db.name} binned {db.shape} pixel {db.pixel*1e6:.0f} um | dead bins {db.dead.sum()}/{db.dead.size}")

rng = np.random.default_rng(42)
t0 = time.perf_counter()
N = 20
stats = []
for i in range(N):
    s = synth_sample(rng, cfg, db)
    img, params, aux = s["img"], s["params"], s["aux"]
    stats.append((s["phase"], img.max(), img.mean(), s["n_reject"], aux[1]))
    if i < 3:
        print(f"[{i}] {s['phase']:>7s} dist={params[0]:.3f}m poni=({params[1]*1e3:.1f},{params[2]*1e3:.1f})mm "
              f"rot=({params[3]:+.3f},{params[4]:+.3f},{params[5]:+.3f}) "
              f"| max={img.max():6d} mean={img.mean():7.1f} vis={int(aux[1])} rej={s['n_reject']} bs={aux[4]:.0f}px")
ms = (time.perf_counter() - t0) / N * 1e3
print(f"\n--- {N} samples, {ms:.1f} ms/张 -> 50k 单进程 {50_000*ms/1e3/60:.1f} min ---")

ph = [s[0] for s in stats]
print("phase 分布:", {p: ph.count(p) for p in set(ph)})
mx = np.array([s[1] for s in stats]); mn = np.array([s[2] for s in stats])
print(f"max 计数: min={mx.min():.0f} med={np.median(mx):.0f} | mean 计数: med={np.median(mn):.1f}")
vis = np.array([s[4] for s in stats])
print(f"可见环数: min={vis.min():.0f} med={np.median(vis):.0f} max={vis.max():.0f} (config>=2)")
assert all(v >= 2 for v in vis), "存在可见环<2的样本！"
assert mx.min() > 100, "存在几乎全黑的样本！"
print("\n✅ 冒烟测试通过")
