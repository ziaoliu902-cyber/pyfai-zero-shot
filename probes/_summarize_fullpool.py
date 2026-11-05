# -*- coding: utf-8 -*-
"""汇总 practice_fullpool 逐样本 CSV → 全量分档对比表（含配对检验）。"""
import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "probes", "results")
PX_MM = 0.344  # 像素 344 μm


def load(fp):
    rows = list(csv.DictReader(open(fp, encoding="utf-8")))
    d = {}
    for r in rows:
        key = (r["model"], r["recipe"], r["idx"])
        d[key] = r
    return rows, d


def stats(e_dist, poni_mm, rot2):
    return dict(
        n=len(e_dist),
        dist_med=float(np.median(e_dist)),
        dist_mae=float(np.mean(e_dist)),
        lt05=float((e_dist < 0.5).mean() * 100),
        poni_px_med=float(np.median(poni_mm / PX_MM)),
        poni_px_p75=float(np.percentile(poni_mm / PX_MM, 75)),
        poni_px_p90=float(np.percentile(poni_mm / PX_MM, 90)),
        rot2_med=float(np.median(rot2)),
        stuck=float((e_dist > 5).mean() * 100),
    )


def main():
    fps = sorted(glob.glob(os.path.join(RES, "practice_fullpool_*.csv")))
    if len(sys.argv) > 1:
        fps = [f for f in fps if sys.argv[1] in f]
    fp = fps[-1]
    rows, _ = load(fp)
    print(f"源文件 {fp}   行数 {len(rows)}")
    combos = sorted({(r["model"], r["recipe"]) for r in rows})
    idxs = sorted({r["idx"] for r in rows})
    print(f"样本数 {len(idxs)}  组合 {len(combos)}\n")

    hdr = (f"{'模型':26s}{'配方':6s}{'N':>5s}{'dist中位':>9s}{'distMAE':>9s}{'<0.5mm':>8s}"
           f"{'poni中位px':>11s}{'poni p75':>10s}{'poni p90':>10s}{'rot2中位':>9s}{'卡住%':>7s}")
    print(hdr)
    print("-" * len(hdr))
    store = {}
    for c in combos:
        rs = [r for r in rows if (r["model"], r["recipe"]) == c]
        d = np.array([float(r["dist_mm"]) for r in rs])
        p1 = np.array([float(r["poni1_mm"]) for r in rs])
        p2 = np.array([float(r["poni2_mm"]) for r in rs])
        r2 = np.array([float(r["rot2_mrad"]) for r in rs])
        s = stats(d, (p1 + p2) / 2, r2)
        store[c] = (d, (p1 + p2) / 2, r2, {r["idx"] for r in rs})
        print(f"{c[0]:26s}{c[1]:6s}{s['n']:5d}{s['dist_med']:9.3f}{s['dist_mae']:9.3f}"
              f"{s['lt05']:7.0f}%{s['poni_px_med']:11.2f}{s['poni_px_p75']:10.2f}"
              f"{s['poni_px_p90']:10.2f}{s['rot2_med']:9.3f}{s['stuck']:6.0f}%")

    # 配对比较（同 idx）
    print("\n配对比较（同一样本集合，符号检验）")
    keys = list(store)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            common = store[a][3] & store[b][3]
            if not common:
                continue
            def vec(c, col):
                return np.array([float(r[col]) for r in rows
                                 if (r["model"], r["recipe"]) == c and r["idx"] in common])
            for col, name, better in (("dist_mm", "dist", "min"), ("poni2_mm", "poni", "min"),
                                      ("rot2_mrad", "rot2", "min")):
                va, vb = vec(a, col), vec(b, col)
                if better == "min":
                    w = int((va < vb).sum())
                else:
                    w = int((va > vb).sum())
                n = len(va)
                # 符号检验双侧 p（正态近似）
                z = (w - n / 2) / (np.sqrt(n) / 2)
                from math import erfc
                p = erfc(abs(z) / np.sqrt(2))
                print(f"  {a[0].split('/')[-1]:12s}{a[1]} vs {b[0].split('/')[-1]:12s}{b[1]} "
                      f"| {name}: {w}/{n} 胜  p≈{p:.3f}")


if __name__ == "__main__":
    main()
