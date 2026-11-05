# -*- coding: utf-8 -*-
"""gen_widenear4.py：真实帧邻域致密卷（治 v4w2 的条件凹槽）。

诊断链（10-31）：σq 扫描显示 v4w2 在固定几何下 dist 读数随 σq 摆 ±25~50mm，
widenear2 的 12k 撒在 k(1.9,2.9)×λ(0.5,2.3Å) 巨域 → 真实帧角落
(k2.42, λ0.97, dist0.15~0.45, σq0.015~0.03) 条件密度仅 ~2 张 → 凹槽残留。

v4 配方（全域消相关 + 邻域致密）：
  k      连续 U(2.30, 2.55)   ← 真实帧 k=2.4244 邻域
  λ      U(0.92, 1.02) Å     ← 真实 λ=0.9724 邻域
  dist   (0.10, 0.70) m      ← 真实 0.15~0.45 + 余量
  σq     连续 log-U(0.002, 0.06)
  相     50% CeO2，其余随机
输出 data/train_widenear4；进度落 widenear4_live.log。
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

N_FULL = 20000
SEED = 20261037
DIST = (0.10, 0.70)
K_RANGE = (2.30, 2.55)
SIG = (0.002, 0.06)
LAM = (0.92e-10, 1.02e-10)
CEO2_FRAC = 0.5
SHARD = 256
LOG = os.path.join(ROOT, "probes", "results", "widenear4_live.log")


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
    out = os.path.join(ROOT, "data", "train_widenear4")
    os.makedirs(out, exist_ok=True)
    total = N_FULL
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

    imgs, params, aux = [], [], []
    shard_i = len([f for f in os.listdir(out)
                   if f.startswith("shard_") and f.endswith(".npz")])
    rejects = n = 0
    t0 = time.time()
    print(f"[widenear4 点火 {time.strftime('%H:%M:%S')}] N={total} "
          f"dist{DIST} k连续({K_RANGE[0]},{K_RANGE[1]}) "
          f"σq连续{SIG} λ({LAM[0]*1e10:.2f},{LAM[1]*1e10:.2f}Å) "
          f"CeO2占比{CEO2_FRAC}", flush=True)
    while n < total:
        k = float(rng.uniform(*K_RANGE))
        db = db_for(k)
        ph = ceo2 if rng.random() < CEO2_FRAC else None
        try:
            s = synth_sample(rng, cfg, db, phase_idx=ph, pixel_scale=k)
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
        if n % 500 == 0:
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
                lam_range_A=[LAM[0] * 1e10, LAM[1] * 1e10],
                ceo2_frac=CEO2_FRAC,
                status="done",
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"完成 {n} 张（拒绝 {rejects}）→ {out}", flush=True)


if __name__ == "__main__":
    main()
