# -*- coding: utf-8 -*-
"""AgBh 远距小灶 v2：data/train_agbhfar2（10-30 晨间首针）。

v1 教训（10-29 深夜）：λ log-U(0.5,1.2) 撒太散，73% 短 λ，病灶
（长 λ 远距，真实 regime 1.0Å/1.58m）没喂饱 → 偏差 +440 未愈。
v2 药方：λ 收窄 (0.9,1.3)、dist 收窄 (1.2,2.5)，直击病灶。
此针同时是判别实验：偏差塌=数据问题；不动=物理极限（转精修）。
"""
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))

from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_detector, synth_sample

N_FULL = 4000
SEED = 20261034
DIST = (1.2, 2.5)
LAM = (9e-11, 1.3e-10)
BROAD_FRAC = 0.18
BROAD_RANGE = (0.05, 0.6)
SHARD = 256


def main():
    out = os.path.join(ROOT, "data", "train_agbhfar2")
    os.makedirs(out, exist_ok=True)
    total = N_FULL
    cfg = SimConfig()
    cfg.shard_size = SHARD
    cfg.dist_range = DIST
    cfg.wavelength_range = LAM
    cfg.peak_sigma_broad_frac = BROAD_FRAC
    cfg.peak_sigma_broad_range = BROAD_RANGE
    db = get_detector("Pilatus_1M", 2)
    ph_i = PHASE_NAMES.index("AgBh")
    rng = np.random.default_rng(SEED)

    imgs, params, aux = [], [], []
    shard_i = len([f for f in os.listdir(out)
                   if f.startswith("shard_") and f.endswith(".npz")])
    rejects = n = 0
    t0 = time.time()
    while n < total:
        try:
            s = synth_sample(rng, cfg, db, phase_idx=ph_i, pixel_scale=1.0)
        except RuntimeError:
            rejects += 1
            continue
        imgs.append(s["img"]); params.append(s["params"]); aux.append(s["aux"])
        n += 1
        if len(imgs) >= SHARD:
            np.savez(os.path.join(out, f"shard_{shard_i:05d}.npz"),
                     imgs=np.stack(imgs), params=np.stack(params),
                     aux=np.stack(aux))
            shard_i += 1
            imgs.clear(); params.clear(); aux.clear()
        if n % 400 == 0:
            rate = n / max(time.time() - t0, 1e-9)
            print(f"{n}/{total} {(time.time()-t0)/60:.1f}min {rate:.1f}/s "
                  f"ETA {(total-n)/max(rate,1e-9)/60:.0f}min 拒绝={rejects}",
                  flush=True)
    if imgs:
        np.savez(os.path.join(out, f"shard_{shard_i:05d}.npz"),
                 imgs=np.stack(imgs), params=np.stack(params),
                 aux=np.stack(aux))
        shard_i += 1
    meta = dict(config=cfg.to_dict(), total=total, actual=n, rejects=rejects,
                phase_names=PHASE_NAMES, forced_phase="AgBh",
                pixel_scales=[1.0], status="done",
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"完成 {n} 张（拒绝 {rejects}）→ {out}", flush=True)


if __name__ == "__main__":
    main()
