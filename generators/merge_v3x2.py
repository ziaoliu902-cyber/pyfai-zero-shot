# -*- coding: utf-8 -*-
"""合仓 v3x2 = train_v3x（63k 全量）+ train_agbhfar2（4k AgBh 远距加密）。

硬链接 v3x 分片（同卷零拷贝，Dataset 需分片做行索引）+
拷贝 agbhfar2 分片（重命名接续编号）→ 新打包 imgs/params/aux。
行序 = 分片名排序序，与 v3x 原 imgs.npy 序一致（其本就是 pack 产物）。
"""
import json
import os
import shutil
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_A = os.path.join(ROOT, "data", "train_v3x")          # 主体（借分片）
SRC_B = os.path.join(ROOT, "data", "train_agbhfar2")     # 加密小灶（拷贝）
DST = os.path.join(ROOT, "data", "train_v3x2")


def main():
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)
    for f in os.listdir(DST):
        os.remove(os.path.join(DST, f))

    sa = sorted(f for f in os.listdir(SRC_A)
                if f.startswith("shard_") and f.endswith(".npz"))
    sb = sorted(f for f in os.listdir(SRC_B)
                if f.startswith("shard_") and f.endswith(".npz"))
    na = len(sa)
    print(f"v3x 分片 {na} 个（硬链接），agbhfar2 分片 {len(sb)} 个（拷贝）",
          flush=True)
    for f in sa:
        os.link(os.path.join(SRC_A, f), os.path.join(DST, f))
    for j, f in enumerate(sb):
        shutil.copy2(os.path.join(SRC_B, f),
                     os.path.join(DST, f"shard_{na + j:05d}.npz"))
        if (j + 1) % 8 == 0:
            print(f"  小灶分片 {j+1}/{len(sb)}", flush=True)

    # 行数与 aux 列数（以分片为准）
    n_a = n_b = 0
    ncol_aux = None
    for f in sa:
        with np.load(os.path.join(SRC_A, f)) as z:
            n_a += len(z["params"])
    for f in sb:
        with np.load(os.path.join(SRC_B, f)) as z:
            n_b += len(z["params"])
            ncol_aux = z["aux"].shape[1]
    N = n_a + n_b
    print(f"行数 {N}（{n_a}+{n_b}），打包 {N*521*490*2/1e9:.1f}GB…", flush=True)

    ia = np.load(os.path.join(SRC_A, "imgs.npy"), mmap_mode="r")
    assert ia.shape[0] == n_a, f"v3x imgs 行数 {ia.shape[0]} != 分片 {n_a}"
    imgs = np.lib.format.open_memmap(
        os.path.join(DST, "imgs.npy"), mode="w+", dtype=np.uint16,
        shape=(N, ia.shape[1], ia.shape[2]))
    imgs[:n_a] = ia[:]
    imgs.flush()
    params = np.empty((N, 6), np.float32)
    aux = np.empty((N, ncol_aux), np.float32)
    params[:n_a] = np.load(os.path.join(SRC_A, "params.npy"))
    aux[:n_a] = np.load(os.path.join(SRC_A, "aux.npy"))
    i = n_a
    for f in sb:
        with np.load(os.path.join(SRC_B, f)) as z:
            n = len(z["params"])
            imgs[i:i + n] = z["imgs"].astype(np.uint16)
            params[i:i + n] = z["params"]
            aux[i:i + n] = z["aux"]
            i += n
        if (i - n_a) % (256 * 8) == 0:
            imgs.flush()
    imgs.flush()
    np.save(os.path.join(DST, "params.npy"), params)
    np.save(os.path.join(DST, "aux.npy"), aux)

    base = json.load(open(os.path.join(SRC_A, "meta.json"), encoding="utf-8"))
    json.dump(dict(config=base["config"], phase_names=base.get("phase_names"),
                   total=N, shards=na + len(sb),
                   mix=dict(v3x=n_a, agbhfar2=n_b),
                   created=time.strftime("%Y-%m-%d %H:%M:%S"),
                   status="merged-hardlink"),
              open(os.path.join(DST, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"完成 train_v3x2：{N} 行，{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
