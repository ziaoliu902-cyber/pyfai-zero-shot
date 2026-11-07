# -*- coding: utf-8 -*-
"""_b7_gapmask_probe.py:B7 模块缝自动 mask·判决探针(2026-11-07)。

问题:Pilatus/Eiger 模块缝(恒 0 行/列带)污染径向谱;用户手做 mask 繁琐。
方案:行列中位能量谱 → 恒暗带(中位 < 阈×全局中位)聚类成带;带内像素
恒 0 比例佐证(死缝 vs 弱衍射区)。

判决口径:
  1) 检出的缝带是否与探测器几何常识吻合(Pilatus2M:横缝 ~17px 周期×多道
     +上下边 2 道;Eiger9M:3×3 大缝;Pilatus300k:1×1 无横缝——阴性对照);
  2) 缝带内像素死 0 比例(真缝 ≈100% 零;织构暗区不会恒 0);
  3) 误报率:Frelon(ceria,连续器件)必须零检出。
"""
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")


def detect_gaps(img, rel=0.05, min_w=2, max_w=80):
    """行列恒暗带检测。返回 (row_bands, col_bands, stats)。

    rel:行/列中位 < rel×全局中位 → 暗;连续暗线聚成带(min_w~max_w px)。
    """
    img = np.asarray(img, dtype=np.float64)
    gm = float(np.median(img))
    if gm <= 0:
        return [], [], dict(global_med=gm)
    row_med = np.median(img, axis=1)
    col_med = np.median(img, axis=0)
    bands = []
    for ax, med in (("row", row_med), ("col", col_med)):
        dark = med < rel * gm
        # 聚类连续暗线
        bl, i = [], 0
        n = len(dark)
        while i < n:
            if dark[i]:
                j = i
                while j + 1 < n and dark[j + 1]:
                    j += 1
                if min_w <= (j - i + 1) <= max_w:
                    bl.append((i, j - i + 1))
                i = j + 1
            else:
                i += 1
        bands.append((ax, bl, dark))
    row_bands = bands[0][1]
    col_bands = bands[1][1]
    stats = dict(global_med=gm,
                 row_dark_frac=float(bands[0][2].mean()),
                 col_dark_frac=float(bands[1][2].mean()))
    return row_bands, col_bands, stats


def band_stats(img, i0, w, axis, gm):
    """带内统计:死0比例 + P95(判据核心:真缝无信号穿带)。"""
    sl = (slice(i0, i0 + w), slice(None)) if axis == 0 else \
         (slice(None), slice(i0, i0 + w))
    b = np.asarray(img)[sl]
    return float((b <= 0).mean()), float(np.percentile(b, 95)) / max(gm, 1e-9)


def probe(path, expect):
    img = fabio.open(path).data.astype(np.float64)
    rb, cb, st = detect_gaps(img)
    gm = st["global_med"]
    rows = []
    n_true = n_fake = 0
    for axis, bands in ((0, rb), (1, cb)):
        for i0, w in bands:
            z, p95 = band_stats(img, i0, w, axis, gm)
            # 真缝判据:带内 P95 < 20% 全局中位(无信号穿带;织构暗带
            # 必有环穿过);Pilatus CBF 缝值=-1 → 死0(z≤0)≈1 双保险
            true_gap = (p95 < 0.20) or (z > 0.98)
            if true_gap:
                n_true += 1
            else:
                n_fake += 1
            rows.append((i0, w, z, p95, true_gap))
    tot = sum(w for _, w in rb) + sum(w for _, w in cb)
    print(f"\n{os.path.basename(path)}  [{expect}]  {img.shape} "
          f"全局中位={gm:.1f}")
    print(f"  行暗带 {len(rb)} 条 {rb[:5]}{'…' if len(rb) > 5 else ''}")
    print(f"  列暗带 {len(cb)} 条 {cb[:5]}{'…' if len(cb) > 5 else ''}")
    for i0, w, z, p95, tg in rows[:8]:
        print(f"    带@{i0:5} w={w:2} 死0={z:.2f} P95/全局={p95:.3f} "
              f"{'真缝' if tg else '误报(织构)'}")
    print(f"  → 真缝={n_true} 误报={n_fake} mask像素={tot}")
    return len(rb), len(cb), n_true, n_fake, tot


print("=" * 70)
print("B7 模块缝自动检测·判决探针")
print("=" * 70)
res = []
# 阳性:Pilatus2M(横缝多道)、Eiger4M、Eiger9M(大缝)、Pilatus1M、Pilatus6M
res.append(("P2M",) + tuple(probe(ROOT + r"\data\real\silx_zoo\Pilatus2M.cbf",
                                  "阳性:横缝周期道+边道")))
res.append(("P1M",) + tuple(probe(ROOT + r"\data\real\silx_zoo\Pilatus1M.edf",
                                  "阳性:单横缝")))
res.append(("E4M",) + tuple(probe(ROOT + r"\data\real\silx_zoo\Eiger4M.edf",
                                  "阳性:Eiger 竖缝+横缝")))
res.append(("E9M",) + tuple(probe(ROOT + r"\data\real\silx_zoo\Eiger9M.h5",
                                  "阳性:Eiger 大缝")))
# 阴性:ceria=Frelon 连续器件;Pilatus300k 单模块无横缝
res.append(("P300k",) + tuple(probe(ROOT + r"\data\real\Pilatus300k.edf",
                                    "阴性:单模块,只边缘")))
res.append(("ceria",) + tuple(probe(
    ROOT + r"\data\real\MX_ceria\ceria_200_1_0001.cbf",
    "阴性:Frelon 连续器件,必须零真缝")))

print("\n" + "=" * 70)
print("判决:真缝=暗带且(带内P95<20%全局中位 或 死0>98%);织构暗带不mask")
for name, nrb, ncb, nz, nf, tot in res:
    print(f"  {name:6} 行带{nrb:3} 列带{ncb:3} 真缝{nz:3} 误报{nf:3} "
          f"maskpx={tot}")
