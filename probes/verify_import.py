# -*- coding: utf-8 -*-
"""落盘验证：torchfai 包可导入、单样本合成耗时与基本健康度（跑完即弃）。"""
import time
import numpy as np
import torchfai
from torchfai.config import SimConfig
from torchfai.simulator import get_detector, synth_sample

print("torchfai", torchfai.__version__)
cfg = SimConfig()
db = get_detector(cfg.detector_name, cfg.binning)
print("detector:", db.name, db.shape, f"pixel={db.pixel*1e6:.0f}um", f"dead={db.dead.mean()*100:.1f}%")

rng = np.random.default_rng(42)
t0 = time.perf_counter()
for _ in range(5):
    synth_sample(rng, cfg, db)
dt = (time.perf_counter() - t0) / 5 * 1000

s = synth_sample(rng, cfg, db, phase_idx=0)
img, aux = s["img"], s["aux"]
print(f"{dt:.1f} ms/img | shape={img.shape} dtype={img.dtype} max={img.max()} "
      f"nonzero={np.count_nonzero(img)/img.size*100:.0f}% "
      f"n_vis={aux[1]:.0f} qmax={aux[2]:.3f} scale={aux[3]:.0f} bs_r_px={aux[4]:.1f} "
      f"rejects={s['n_reject']} phase={s['phase']}")
print("params:", np.round(s["params"], 5))
