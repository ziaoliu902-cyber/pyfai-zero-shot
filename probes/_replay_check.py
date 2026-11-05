# -*- coding: utf-8 -*-
"""_replay_check.py：重放确定性校验——前 5 个接受样本的图像哈希。
打补丁前跑一次取基准；打补丁后再跑必须逐位一致（证明记录 σq 不动随机流）。
"""
import hashlib
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

SEED = 20261037
K_RANGE = (2.30, 2.55)
SIG = (0.002, 0.06)
LAM = (0.92e-10, 1.02e-10)
DIST = (0.10, 0.70)
CEO2_FRAC = 0.5

cfg = SimConfig()
cfg.dist_range = DIST
cfg.wavelength_range = LAM
cfg.peak_sigma_range = SIG
cfg.peak_sigma_broad_frac = 0.0
rng = np.random.default_rng(SEED)
db_cache = {}
ceo2 = PHASE_NAMES.index("CeO2")


def db_for(k):
    key = round(k, 3)
    if key not in db_cache:
        db_cache[key] = get_scaled_detector("Pilatus_1M", 2, k)
    return db_cache[key]


got = 0
while got < 5:
    k = float(rng.uniform(*K_RANGE))
    db = db_for(k)
    ph = ceo2 if rng.random() < CEO2_FRAC else None
    try:
        s = synth_sample(rng, cfg, db, phase_idx=ph, pixel_scale=k)
    except RuntimeError:
        continue
    h = hashlib.sha1(np.ascontiguousarray(s["img"]).tobytes()).hexdigest()[:16]
    sig = getattr(s, "get", lambda *_: None)("sigmas")
    print(f"#{got} k={k:.4f} ph={ph} img_sha={h} sigmas={sig}")
    got += 1

with np.load(r"data\train_widenear4\shard_00000.npz") as z:
    for i in range(5):
        h = hashlib.sha1(np.ascontiguousarray(z["imgs"][i]).tobytes()).hexdigest()[:16]
        print(f"shard0[{i}] img_sha={h}")
