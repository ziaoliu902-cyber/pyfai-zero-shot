# -*- coding: utf-8 -*-
"""gen_sigq_replay.py：确定性重放 widenear4 全程 → 精确 σq 标签（20000 行）。

原理：gen_widenear4 种子驱动，同代码同种子重放 → 图像逐位复现
（_replay_check 已验前 5 张哈希一致）；simulator 已打流中性记录补丁，
每样本导出可见环 σq 中位数。不落图，只收标签。

输出：probes/results/widenear4_sigq.npy (20000,) float32 + _stats.json
对应 train_v4w3 行区间 [79000, 99000)（shard_00311..00389，生成序=行序）。
"""
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

SEED = 20261037
N_FULL = 20000
DIST = (0.10, 0.70)
K_RANGE = (2.30, 2.55)
SIG = (0.002, 0.06)
LAM = (0.92e-10, 1.02e-10)
CEO2_FRAC = 0.5
OUT = os.path.join(ROOT, "probes", "results", "widenear4_sigq.npy")
LOG = os.path.join(ROOT, "probes", "results", "sigq_replay_live.log")


class _Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.s = sys.stdout

    def write(self, s):
        self.s.write(s)
        self.f.write(s)
        return len(s)

    def flush(self):
        self.s.flush()
        self.f.flush()


def main():
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

    sigs_out = []
    rejects = 0
    t0 = time.time()
    print(f"[σq 重放点火 {time.strftime('%H:%M:%S')}] N={N_FULL}", flush=True)
    while len(sigs_out) < N_FULL:
        k = float(rng.uniform(*K_RANGE))
        db = db_for(k)
        ph = ceo2 if rng.random() < CEO2_FRAC else None
        try:
            s = synth_sample(rng, cfg, db, phase_idx=ph, pixel_scale=k)
        except RuntimeError:
            rejects += 1
            continue
        sigs_out.append(float(np.median(s["sigmas"])))
        n = len(sigs_out)
        if n % 1000 == 0:
            rate = n / max(time.time() - t0, 1e-9)
            print(f"{n}/{N_FULL} {(time.time()-t0)/60:.1f}min "
                  f"ETA {(N_FULL-n)/max(rate,1e-9)/60:.0f}min 拒绝={rejects}",
                  flush=True)
    arr = np.asarray(sigs_out, np.float32)
    np.save(OUT, arr)
    with open(OUT.replace(".npy", "_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(n=len(arr), rejects=rejects,
                       min=float(arr.min()), max=float(arr.max()),
                       median=float(np.median(arr))), f, indent=1)
    print(f"完成：N={len(arr)} 拒绝={rejects} "
          f"σq∈[{arr.min():.4f},{arr.max():.4f}] 中位{np.median(arr):.4f} "
          f"总耗时{(time.time()-t0)/60:.0f}min", flush=True)


if __name__ == "__main__":
    sys.stdout = _Tee(LOG)
    main()
