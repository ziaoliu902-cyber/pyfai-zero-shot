# -*- coding: utf-8 -*-
"""验尸宽域粗解偏差：v3x 各段 (dist, k) 覆盖 vs 真实帧需求。"""
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
m = json.load(open(ROOT + r"\data\train_v3x\meta.json", encoding="utf-8"))


def walk(d, path=""):
    if isinstance(d, dict):
        keys = set(d)
        if {"n", "dist_range"} <= keys or {"count", "dist_range"} <= keys:
            n = d.get("n", d.get("count"))
            dist = d.get("dist_range")
            k = d.get("k_range", d.get("pixel_scale_range", "1(默认)"))
            ph = d.get("phase", d.get("phases", "?"))
            print(f"{path:28s} n={n:<6} dist={dist} k={k} 相={ph}")
            return
        for kk, vv in d.items():
            walk(vv, f"{path}/{kk}" if path else kk)
    elif isinstance(d, list):
        for i, vv in enumerate(d):
            walk(vv, f"{path}[{i}]")


walk(m)
print()
# 真实宽帧的需求区间
print("真实宽帧需求: dist=(0.15,0.45)m  k≈2.425  相=CeO2  "
      "亮度=真实曝光(偏暗50~130×,靠match_gain)")
# 顺带统计 aux 里 k 与 dist 的联合分布（wide 段）
try:
    import csv
    rows = list(csv.DictReader(open(ROOT + r"\data\train_v3x\index.csv",
                                    encoding="utf-8")))
    print(f"\nindex.csv 共 {len(rows)} 行，列: {list(rows[0].keys())[:10]}")
except Exception as e:
    print("index 读取:", e)
try:
    aux = np.load(ROOT + r"\data\train_v3x\aux.npy", mmap_mode="r")
    print("aux.npy shape:", aux.shape)
except Exception as e:
    print("aux 读取:", e)
