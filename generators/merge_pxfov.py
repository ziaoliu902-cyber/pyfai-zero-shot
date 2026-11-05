# -*- coding: utf-8 -*-
"""合并 px+宽峰补丁仓与 lam_v1 重放 → data/train_px1（7 列 aux）。

构成：train_pxfov1（28k，k 六档+宽峰分层，7 列 aux）全量
    + train_lam_v1 训练分裂（88%）内前 12000 行重放（防遗忘，
      aux 补第 7 列 k=1；只用 train 分裂——val/test 分裂零沾染，
      与 TorchFAIDataset 同种子同比例的确定性划分复算）
然后 dataset.pack(root) 打平（mmap 随机访问）。

血缘：pxfov1 seed=20261029 k档[1,1.26,1.58,2,2.5,3] broad 18%；
      lam_v1 seed=20261027 λ(0.5,2.3)Å；重放行 = lam_v1 训练分裂前 12k。
"""
import json
import os
import shutil
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchfai.dataset import pack

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_NEW = os.path.join(ROOT, "data", "train_pxfov1")
SRC_OLD = os.path.join(ROOT, "data", "train_lam_v1")
DST = os.path.join(ROOT, "data", "train_px1")
REPLAY_N = 12000
SHARD = 256


def main():
    assert os.path.isdir(SRC_NEW), f"缺 {SRC_NEW}（先跑 gen_pxfov.py）"
    os.makedirs(DST, exist_ok=True)

    # ---- 1. 拷贝新仓分片（已是 7 列 aux）----
    new_shards = sorted(f for f in os.listdir(SRC_NEW)
                        if f.startswith("shard_") and f.endswith(".npz"))
    n_new = 0
    for i, fn in enumerate(new_shards):
        shutil.copy2(os.path.join(SRC_NEW, fn),
                     os.path.join(DST, f"shard_{i:05d}.npz"))
        with np.load(os.path.join(SRC_NEW, fn)) as z:
            n_new += len(z["params"])
    print(f"新仓分片 {len(new_shards)} 个共 {n_new} 行已拷贝", flush=True)

    # ---- 2. 重放选行：lam_v1 训练分裂（seed0/0.88-0.06-0.06）前 12k ----
    old_shards = sorted(f for f in os.listdir(SRC_OLD)
                        if f.startswith("shard_") and f.endswith(".npz"))
    lens = []
    for fn in old_shards:
        with np.load(os.path.join(SRC_OLD, fn)) as z:
            lens.append(int(len(z["params"])))
    offs = np.concatenate([[0], np.cumsum(lens)])
    total = int(offs[-1])
    idx = np.random.default_rng(0).permutation(total)
    sel = np.sort(idx[:int(total * 0.88)])[:REPLAY_N]     # 训练分裂前 12k
    print(f"lam_v1 总 {total}，重放 {len(sel)} 行（训练分裂内）", flush=True)

    shard_i = len(new_shards)
    buf_i, buf_img, buf_p, buf_a = 0, [], [], []
    for g in sel:
        si = int(np.searchsorted(offs, g, side="right") - 1)
        row = int(g - offs[si])
        with np.load(os.path.join(SRC_OLD, old_shards[si])) as z:
            buf_img.append(z["imgs"][row])
            buf_p.append(z["params"][row])
            a6 = z["aux"][row]
            buf_a.append(np.concatenate([a6, [1.0]]).astype(np.float32))
        buf_i += 1
        if buf_i == SHARD:
            np.savez(os.path.join(DST, f"shard_{shard_i:05d}.npz"),
                     imgs=np.stack(buf_img), params=np.stack(buf_p),
                     aux=np.stack(buf_a))
            shard_i += 1
            buf_i, buf_img, buf_p, buf_a = 0, [], [], []
            print(f"重放分片 {shard_i}（累计 {shard_i * SHARD}）", flush=True)
    if buf_img:
        np.savez(os.path.join(DST, f"shard_{shard_i:05d}.npz"),
                 imgs=np.stack(buf_img), params=np.stack(buf_p),
                 aux=np.stack(buf_a))
        shard_i += 1

    # ---- 3. meta：pxfov1 配置 + 血缘注记 ----
    with open(os.path.join(SRC_NEW, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["merged"] = dict(
        new_rows=n_new, replay_rows=int(len(sel)),
        replay_source="train_lam_v1(train-split first 12k, seed0/0.88)",
        created=os.path.basename(DST), status="merged")
    with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"meta 落盘；合计 {n_new + len(sel)} 行 → 打包…", flush=True)

    pack(DST)
    print("完成：train_px1 可训练（--data data/train_px1 --lam-cond --px-cond）")


if __name__ == "__main__":
    main()
