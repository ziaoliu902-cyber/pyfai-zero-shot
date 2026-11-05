# -*- coding: utf-8 -*-
"""迭代预算消融（论文用）：400 vs 1200 迭代，同卷配对比较。

数据源：
  runs/refine_v2/results.csv    快速档 400 迭代（全量 3900）
  runs/refine_final/results.csv 精细档 1200 迭代（2313 张后停止——证据已饱和）

结论（2026-10-25）：配对中位差≈0，长尾不变，卡住率 6.9%→6.8%，
1200 档逐卷仅 44% 更准 → 精修 ~400 迭代即收敛，瓶颈在 NN 初值而非计算量。
用法：python probes/iter_ablation.py
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(fp):
    d = {}
    for l in open(fp, encoding="utf-8").readlines()[1:]:
        if l.strip():
            r = l.strip().split(",")
            d[int(r[0])] = np.array([float(x) for x in r[2:]])
    return d


def main():
    fast = load(os.path.join(ROOT, "runs/refine_v2/results.csv"))
    fine = load(os.path.join(ROOT, "runs/refine_final/results.csv"))
    ks = sorted(set(fast) & set(fine))
    print(f"配对样本 N={len(ks)}")
    rows = []
    for j, nm, unit in ((7, "dist", "mm"), (8, "poni1", "mm"), (10, "rot1", "mrad")):
        fa = np.array([fast[k][j] for k in ks])
        fi = np.array([fine[k][j] for k in ks])
        rows.append((nm, np.median(fa), np.median(fi),
                     np.percentile(fa, 90), np.percentile(fi, 90)))
        print(f"{nm:6s}  中位 {np.median(fa):8.3f} → {np.median(fi):8.3f} {unit}   "
              f"P90 {np.percentile(fa,90):7.3f} → {np.percentile(fi,90):7.3f}")
    fa = np.array([fast[k][7] for k in ks]); fi = np.array([fine[k][7] for k in ks])
    print(f"卡住(>5mm): {(fa>5).mean()*100:.1f}% → {(fi>5).mean()*100:.1f}%")
    print(f"逐卷胜负: 精细档更准比例 {(fi<fa).mean()*100:.0f}%（<50% 即无优势）")
    print("\n论文表述建议：")
    print("  \"Refinement converges within ~400 iterations; tripling the budget")
    print("   yields no paired improvement (Δmedian≈0, stuck-rate unchanged),")
    print("   indicating the residual error stems from initialization, not compute.\"")


if __name__ == "__main__":
    main()
