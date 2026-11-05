# -*- coding: utf-8 -*-
"""AgBh 远距定向小灶：data/train_agbhfar（4k，v3far 收官补刀）。

物理依据（10-29 深夜定案）：远距视窗 q_max 急缩（1.58m/λ1Å → 0.67），
六相中仅 AgBh（首环 0.108）在远距有信息——远距考题物理上只有 AgBh
答得了。v3far 后 AgBh@远距仍错 +451mm → 本卷专治。

配置：相固定 AgBh，dist∈(1.0,3.0) log-U，λ∈(0.5,1.2)Å，k=1，
宽峰 18% 与主仓一致。自写循环带不可行重试（同 gen_v3adds）。
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
SEED = 20261033
DIST = (1.0, 3.0)
LAM = (5e-11, 1.2e-10)
BROAD_FRAC = 0.18
BROAD_RANGE = (0.05, 0.6)
SHARD = 256


def main():
    smoke = "--smoke" in sys.argv
    out = os.path.join(ROOT, "data", "_smoke_agbhfar" if smoke
                       else "train_agbhfar")
    os.makedirs(out, exist_ok=True)
    total = 128 if smoke else N_FULL
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
