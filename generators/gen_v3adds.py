# -*- coding: utf-8 -*-
"""v3 增补卷：data/train_v3adds（10-29 深夜覆盖洞终局的根治数据）。

三个 stratum（宽峰 18% 与 pxfov1 一致）：
  A. far@1M  12000 行：k=1, dist∈(1.0,3.0)m, λ∈(0.5,1.2)Å
     —— 治 sim 实锤的 far@k=1 dist 崩坏（-260~-2026mm）与 poni 装傻先验
  B. k<1      7000 行：k∈{0.55,0.7,0.85}, dist∈(0.1,1.2)m, λ∈(0.5,1.5)Å
     —— 治 alumina 拉满画布布局（小探测器 zoom-to-fill ≡ k<1，物理等价）
  C. 高亮饱和 4000 行：k=1, dist∈(0.5,3.0)m, λ∈(0.5,1.2)Å, 峰值(5e4,3e5)
     —— 治真实 AgBh 245k 饱和平顶环（u16 天然裁剪出饱和带）+ 增益悬崖

自写循环（不走 generate_gui.run_generation）：远距 ~10% 的 (λ,相,dist)
组合本征不可行（无两环可见），run_generation 无外层重试会一张崩全场；
这里 try/except → 重抽，拒绝计数入 meta。

用法：python gen_v3adds.py            # 23k 张 → data/train_v3adds
      python gen_v3adds.py --smoke    # 快检 384 张
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
from torchfai.simulator import get_scaled_detector, synth_sample

BROAD_FRAC = 0.18
BROAD_RANGE = (0.05, 0.6)

STRATA = [
    dict(name="far1M", n=12000, seed=20261030,
         pixel_scales=[1.0], dist=(1.0, 3.0), lam=(5e-11, 1.2e-10)),
    dict(name="ksub1", n=7000, seed=20261031,
         pixel_scales=[0.55, 0.7, 0.85], dist=(0.1, 1.2),
         lam=(5e-11, 1.5e-10)),
    dict(name="bright", n=4000, seed=20261032,
         pixel_scales=[1.0], dist=(0.5, 3.0), lam=(5e-11, 1.2e-10),
         extra=dict(peak_counts_range=(5e4, 3e5))),
]
SHARD = 256


def gen_stratum(st, out, smoke):
    from torchfai.simulator import get_detector
    total = 128 if smoke else st["n"]
    cfg = SimConfig()
    cfg.shard_size = SHARD
    cfg.dist_range = tuple(st["dist"])
    cfg.wavelength_range = tuple(st["lam"])
    cfg.peak_sigma_broad_frac = BROAD_FRAC
    cfg.peak_sigma_broad_range = BROAD_RANGE
    for k, v in st.get("extra", {}).items():
        setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    dbs = {kk: (get_scaled_detector("Pilatus_1M", 2, kk)
                if kk != 1.0 else get_detector("Pilatus_1M", 2))
           for kk in st["pixel_scales"]}
    ks = sorted(dbs)
    rng = np.random.default_rng(st["seed"])

    imgs, params, aux = [], [], []
    shard_i = len([f for f in os.listdir(out)
                   if f.startswith("shard_") and f.endswith(".npz")])
    counts = {p: 0 for p in PHASE_NAMES}
    rejects = n = 0
    t0 = time.time()

    def flush(final=False):
        nonlocal shard_i
        if imgs:
            fp = os.path.join(out, f"shard_{shard_i:05d}.npz")
            np.savez(fp, imgs=np.stack(imgs),
                     params=np.stack(params), aux=np.stack(aux))
            shard_i += 1
            imgs.clear(); params.clear(); aux.clear()
        meta = dict(stratum=st["name"], config=cfg.to_dict(), total=total,
                    actual=n, rejects=rejects, counts=counts,
                    pixel_scales=ks, status="done" if final else "running",
                    created=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(os.path.join(out, f"meta_{st['name']}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"== {st['name']}: N={total} k={ks} dist={st['dist']} "
          f"λ={st['lam'][0]*1e10:.1f}-{st['lam'][1]*1e10:.1f}Å ==", flush=True)
    while n < total:
        kk = ks[rng.integers(len(ks))]
        try:
            s = synth_sample(rng, cfg, dbs[kk], phase_idx=None,
                             pixel_scale=kk)
        except RuntimeError:
            rejects += 1
            continue
        imgs.append(s["img"]); params.append(s["params"]); aux.append(s["aux"])
        counts[s["phase"]] += 1
        rejects += s.get("n_reject", 0)
        n += 1
        if len(imgs) >= SHARD:
            flush()
        if n % 200 == 0:
            rate = n / max(time.time() - t0, 1e-9)
            print(f"[{st['name']}] {n:6d}/{total} {(time.time()-t0)/60:5.1f}min"
                  f" {rate:5.1f}/s ETA {(total-n)/max(rate,1e-9)/60:3.0f}min"
                  f" 拒绝={rejects}", flush=True)
    flush(final=True)
    print(f"[{st['name']}] 完成 {n} 张，拒绝 {rejects}", flush=True)


def main():
    smoke = "--smoke" in sys.argv
    out = os.path.join(ROOT, "data", "_smoke_v3adds" if smoke
                       else "train_v3adds")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for st in STRATA:
        gen_stratum(st, out, smoke)
    print(f"\n全部完成 {(time.time()-t0)/60:.1f}min → {out}", flush=True)


if __name__ == "__main__":
    main()
