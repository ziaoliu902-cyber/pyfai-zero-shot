# -*- coding: utf-8 -*-
"""联合分布：v3x 的 (dist, k) 覆盖 vs 真实宽帧需求 (0.15~0.45m, k2.425)。"""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
m = json.load(open(ROOT + r"\data\train_v3x\meta.json", encoding="utf-8"))
print("meta keys:", list(m.keys()))
for k in ("segments", "config", "provenance", "sources"):
    if k in m:
        v = m[k]
        print(f"\n== {k} ==")
        print(json.dumps(v, ensure_ascii=False, indent=1)[:1200])

aux = np.load(ROOT + r"\data\train_v3x\aux.npy", mmap_mode="r")
k_col = np.asarray(aux[:, 6], dtype=float)
lam = np.asarray(aux[:, 5], dtype=float) if aux.shape[1] >= 6 else None
# dist 无法从 aux 拿（aux 是 λ/k），改从 params.npy
try:
    P = np.load(ROOT + r"\data\train_v3x\params.npy", mmap_mode="r")
    dist = np.asarray(P[:, 0], dtype=float)
    print("\nparams.npy shape:", P.shape)
    wide = k_col > 1.5
    print(f"\n宽域样本(k>1.5): {wide.sum()} 个")
    if wide.any():
        print(f"  dist 范围: {dist[wide].min():.3f} ~ "
              f"{dist[wide].max():.3f} m")
        print(f"  k   范围: {k_col[wide].min():.3f} ~ "
              f"{k_col[wide].max():.3f}")
        # 真实帧需求洞：dist<0.3 且 k>1.5
        hole = wide & (dist < 0.30)
        print(f"  ★洞: dist<0.30m 且 k>1.5 → {hole.sum()} 个"
              f"（真实 150/200/250mm 档就住这）")
        hole2 = wide & (dist < 0.50)
        print(f"  ★洞2: dist<0.50m 且 k>1.5 → {hole2.sum()} 个"
              f"（450mm 档在内）")
        import collections
        h, e = np.histogram(dist[wide], bins=8)
        print("\n  宽域 dist 直方图:")
        for i in range(8):
            bar = "#" * int(60 * h[i] / max(h.max(), 1))
            print(f"   {e[i]:.2f}~{e[i+1]:.2f}m {h[i]:6d} {bar}")
except Exception as ex:
    print("params 读取:", ex)
