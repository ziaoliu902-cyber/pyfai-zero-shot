# -*- coding: utf-8 -*-
"""npz 模式数据集的 params/aux 读取助手（far_patch_v1 等无 imgs.npy 的数据集）。"""
import json
import os

import numpy as np


def npz_params_aux(root):
    """返回 (params[N,6], aux[N,5], meta)，按分片顺序拼接。"""
    meta = json.load(open(os.path.join(root, "meta.json"), encoding="utf-8"))
    shards = sorted(f for f in os.listdir(root)
                    if f.startswith("shard_") and f.endswith(".npz"))
    ps, ax = [], []
    for fn in shards:
        with np.load(os.path.join(root, fn)) as z:
            ps.append(z["params"])
            ax.append(z["aux"])
    return np.concatenate(ps), np.concatenate(ax), meta


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torchfai.phases import PHASE_NAMES

    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p, a, meta = npz_params_aux("data/far_patch_v1")
    d = p[:, 0]
    print(f"far_patch_v1 N={len(p)}  dist_range={meta['config']['dist_range']}")
    for lo, hi, nm in ((0.0, 0.77, "<0.77m"), (0.77, 1.0, "0.77-1.0m"),
                       (1.0, 2.0, "1.0-2.0m"), (2.0, 3.1, ">2.0m")):
        m = (d >= lo) & (d < hi)
        print(f"  {nm:10s} {m.mean()*100:5.1f}%  (N={m.sum()})")
    print(f"  >0.77m {(d>0.77).mean()*100:.1f}%   >1.0m {(d>1.0).mean()*100:.1f}%")
    ph = np.bincount(a[:, 0].astype(int), minlength=6)
    print("  相构成: " + "  ".join(f"{PHASE_NAMES[i]} {ph[i]/len(a)*100:.0f}%" for i in range(6)))
    prac = (a[:, 1] >= 6) & (a[:, 3] >= 10000) & (a[:, 4] < 0)
    print(f"  实际工况画像比例: {prac.mean()*100:.1f}%")

    print("\n--- hard_patch_v1/agbh_far ---")
    p2, a2, m2 = npz_params_aux("data/hard_patch_v1/agbh_far")
    d2 = p2[:, 0]
    print(f"  N={len(p2)} >0.77m {(d2>0.77).mean()*100:.1f}%  >1.0m {(d2>1.0).mean()*100:.1f}%  "
          f"median={np.median(d2):.2f}")
    prac2 = (a2[:, 1] >= 6) & (a2[:, 3] >= 10000) & (a2[:, 4] < 0)
    print(f"  实际工况画像比例: {prac2.mean()*100:.1f}%")
