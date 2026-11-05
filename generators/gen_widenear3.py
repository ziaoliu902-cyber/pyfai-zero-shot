# -*- coding: utf-8 -*-
"""gen_widenear3.py：填缝卷 v3（治真实帧中心死区）。

判决链（10-30 终审复盘）：
  v4w2 已证：σq/k/λ 域覆盖无罪（填平后真实帧偏差 40.1→41.0 纹丝不动）
  移植实验：环强度/背景剖面无罪（孪生+44.6 → 移植后 +44.6 不动）
  物理终审：真实帧中心低信号半径 225~280px，仿真挡光盘 ≤45px ← 病根
  病机：死区吃掉内环 → 环越往中心压（dist 越远）失踪越多 → 模型低估 dist

v3 配方 = v2 全域消相关 + 大遮挡：
  dist   (0.10, 3.00) m
  k      连续 U(1.9, 2.9)
  σq     连续 log-U(0.002, 0.06)
  λ      (0.5, 2.3) Å
  挡光盘 prob=1.0, r ∈ (40, 280) binned px   ← 新增，与 dist 独立采样
输出 data/train_widenear3；进度落 widenear3_live.log。
"""
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_detector, get_scaled_detector, synth_sample

N_FULL = 12000
SEED = 20261037
DIST = (0.10, 3.00)
K_RANGE = (1.9, 2.9)
SIG = (0.002, 0.06)
LAM = (5e-11, 2.3e-10)
SHARD = 256
LOG = os.path.join(ROOT, "probes", "results", "widenear3_live.log")


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

    def reconfigure(self, **kw):
        try:
            self.s.reconfigure(**kw)
        except Exception:
            pass


sys.stdout = _Tee(LOG)


def main():
    out = os.path.join(ROOT, "data", "train_widenear3")
    os.makedirs(out, exist_ok=True)
    total = N_FULL
    cfg = SimConfig()
    cfg.dist_range = DIST
    cfg.wavelength_range = LAM
    cfg.peak_sigma_range = SIG
    cfg.peak_sigma_broad_frac = 0.0
    cfg.beamstop_prob = 1.0                # ← v3 核心：全员大遮挡
    cfg.beamstop_r_range = (40.0, 280.0)   # ← 覆盖真实 225~280px
    rng = np.random.default_rng(SEED)
    db_cache = {}

    def db_for(k):
        key = round(k, 3)
        if key not in db_cache:
            db_cache[key] = (get_detector("Pilatus_1M", 2) if abs(k - 1) < 1e-6
                             else get_scaled_detector("Pilatus_1M", 2, k))
        return db_cache[key]

    imgs, params, aux = [], [], []
    shard_i = len([f for f in os.listdir(out)
                   if f.startswith("shard_") and f.endswith(".npz")])
    rejects = n = 0
    t0 = time.time()
    print(f"[widenear3 点火 {time.strftime('%H:%M:%S')}] N={total} "
          f"dist{DIST} k连续({K_RANGE[0]},{K_RANGE[1]}) σq连续{SIG} "
          f"λ({LAM[0]*1e10:.1f},{LAM[1]*1e10:.1f}Å) "
          f"挡光盘r(40,280)px prob=1", flush=True)
    while n < total:
        k = float(rng.uniform(*K_RANGE))
        db = db_for(k)
        try:
            s = synth_sample(rng, cfg, db, phase_idx=None, pixel_scale=k)
        except RuntimeError:
            rejects += 1
            continue
        imgs.append(s["img"])
        params.append(s["params"])
        aux.append(s["aux"])
        n += 1
        if len(imgs) >= SHARD:
            np.savez(os.path.join(out, f"shard_{shard_i:05d}.npz"),
                     imgs=np.stack(imgs), params=np.stack(params),
                     aux=np.stack(aux))
            shard_i += 1
            imgs.clear(), params.clear(), aux.clear()
        if n % 200 == 0:
            rate = n / max(time.time() - t0, 1e-9)
            print(f"{n}/{total} {(time.time()-t0)/60:.1f}min {rate:.1f}/s "
                  f"ETA {(total-n)/max(rate,1e-9)/60:.0f}min 拒绝={rejects}",
                  flush=True)
    if imgs:
        np.savez(os.path.join(out, f"shard_{shard_i:05d}.npz"),
                 imgs=np.stack(imgs), params=np.stack(params),
                 aux=np.stack(aux))
    meta = dict(config=cfg.to_dict(), total=total, actual=n, rejects=rejects,
                phase_names=PHASE_NAMES,
                k_continuous=list(K_RANGE), sigma_gap_filled=list(SIG),
                occlusion="beamstop prob=1 r(40,280)px",
                status="done",
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"完成 {n} 张（拒绝 {rejects}）→ {out}", flush=True)


if __name__ == "__main__":
    main()
