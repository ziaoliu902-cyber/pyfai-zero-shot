# -*- coding: utf-8 -*-
"""gen_widenear2.py：填缝卷 v2（治 v1 的伪相关毒）。

v1 死因（10-30 验尸复盘）：dist 只采 (0.10,0.52) 近域、k 却连续 (1.9,2.9)
→ 模型学到「k≈2.4/宽环风格 ⇒ 近距」伪相关，真实帧 0.45m 被拽到 -150mm。

v2 配方（关键变化 → 全域消相关）：
  dist   (0.10, 3.00) m    ← 对齐基卷全域，杀 k-距离伪相关
  k      连续 U(1.9, 2.9)  ← 同 v1
  σq     连续 log-U(0.002, 0.06) ← 同 v1（真实 σq≈0.015~0.03 的缝）
  λ      (0.5, 2.3) Å      ← 对齐基卷全域，杀 λ-风格伪相关
  相     六相随机
输出 data/train_widenear2；进度落 widenear2_live.log。
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
SEED = 20261036
DIST = (0.10, 3.00)
K_RANGE = (1.9, 2.9)
SIG = (0.002, 0.06)
LAM = (5e-11, 2.3e-10)
SHARD = 256
LOG = os.path.join(ROOT, "probes", "results", "widenear2_live.log")


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
    out = os.path.join(ROOT, "data", "train_widenear2")
    os.makedirs(out, exist_ok=True)
    total = N_FULL
    cfg = SimConfig()
    cfg.dist_range = DIST
    cfg.wavelength_range = LAM
    cfg.peak_sigma_range = SIG
    cfg.peak_sigma_broad_frac = 0.0
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
    print(f"[widenear2 点火 {time.strftime('%H:%M:%S')}] N={total} "
          f"dist{DIST} k连续({K_RANGE[0]},{K_RANGE[1]}) "
          f"σq连续{SIG} λ({LAM[0]*1e10:.1f},{LAM[1]*1e10:.1f}Å)", flush=True)
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
                status="done",
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"完成 {n} 张（拒绝 {rejects}）→ {out}", flush=True)


if __name__ == "__main__":
    main()
