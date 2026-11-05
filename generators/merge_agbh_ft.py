# -*- coding: utf-8 -*-
"""微训仓：data/train_agbh_ft = AgBh远距 4k + train_v3x 随机重放 8k。

重放防遗忘：纯 AgBh 微训会冲掉六域本事；1:2 掺重放，5-6 个 epoch
低 lr(4e-5) 只允许增量修复。v3x 读取用 mmap 打包镜像（分片已并入）。
"""
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchfai.dataset import pack

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_A = os.path.join(ROOT, "data", "train_agbhfar")
SRC_B = os.path.join(ROOT, "data", "train_v3x")
DST = os.path.join(ROOT, "data", "train_agbh_ft")
REPLAY_N = 8000
SHARD = 256


def main():
    assert os.path.isfile(os.path.join(SRC_A, "meta.json")), "先跑 gen_agbhfar"
    os.makedirs(DST, exist_ok=True)
    for f in os.listdir(DST):
        os.remove(os.path.join(DST, f))

    # 1. AgBh 小灶分片原样拷
    shards_a = sorted(f for f in os.listdir(SRC_A)
                      if f.startswith("shard_") and f.endswith(".npz"))
    i = 0
    for fn in shards_a:
        os.replace if False else None
        import shutil
        shutil.copy2(os.path.join(SRC_A, fn), os.path.join(DST, f"shard_{i:05d}.npz"))
        i += 1
    n_a = sum(len(np.load(os.path.join(DST, f"shard_{j:05d}.npz"))["params"])
              for j in range(i))
    print(f"AgBh 小灶 {n_a} 行（{i} 分片）", flush=True)

    # 2. v3x mmap 随机抽 8k 重放（train 分裂内：seed0/0.88 同款划分）
    mi = np.load(os.path.join(SRC_B, "imgs.npy"), mmap_mode="r")
    mp = np.load(os.path.join(SRC_B, "params.npy"))
    ma = np.load(os.path.join(SRC_B, "aux.npy"))
    total = mi.shape[0]
    perm = np.random.default_rng(0).permutation(total)
    sel = np.sort(perm[:int(total * 0.88)])[:REPLAY_N]
    print(f"v3x 重放 {len(sel)} 行（train 分裂内）", flush=True)

    buf = 0
    for g in sel:
        if buf == 0:
            bi, bp, ba = [], [], []
        bi.append(np.array(mi[g])); bp.append(mp[g]); ba.append(ma[g])
        buf += 1
        if buf == SHARD:
            np.savez(os.path.join(DST, f"shard_{i:05d}.npz"),
                     imgs=np.stack(bi), params=np.stack(bp), aux=np.stack(ba))
            i += 1; buf = 0
            if i % 10 == 0:
                print(f"重放分片 {i}", flush=True)
    if buf:
        np.savez(os.path.join(DST, f"shard_{i:05d}.npz"),
                 imgs=np.stack(bi), params=np.stack(bp), aux=np.stack(ba))
        i += 1

    # 3. meta：继承 v3x config（norm 常量一致性）
    base = json.load(open(os.path.join(SRC_B, "meta.json"), encoding="utf-8"))
    json.dump(dict(config=base["config"], phase_names=base.get("phase_names"),
                   total=n_a + len(sel), shards=i,
                   mix=dict(agbh_far=n_a, v3x_replay=int(len(sel))),
                   created=time.strftime("%Y-%m-%d %H:%M:%S"),
                   status="merged"),
              open(os.path.join(DST, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"合计 {n_a + len(sel)} 行，打包…", flush=True)
    pack(DST)
    print("完成：train_agbh_ft（微训用，--init-from runs/v3far/best.pt）")


if __name__ == "__main__":
    main()
