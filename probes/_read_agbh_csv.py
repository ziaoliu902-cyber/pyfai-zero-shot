# -*- coding: utf-8 -*-
"""读终考 CSV：AgBh 精修明细按 loss 排序。"""
import csv
import glob
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
fp = sorted(glob.glob(r"D:\pyfai-project\probes\results"
                      r"\real_refine_v6_20261030_*.csv"))[-1]
rows = [r for r in csv.DictReader(open(fp, encoding="utf-8"))
        if r["case"].startswith("AgBh")]
rows.sort(key=lambda r: float(r["loss"]))
print(f"文件: {fp}")
print(f"AgBh 精修行数: {len(rows)}")
for r in rows[:10]:
    print(f"{r['start']:>7s}  distΔ={float(r['dist_err_mm']):+8.2f}mm  "
          f"poniΔ=({float(r['poni1_px']):+6.1f},"
          f"{float(r['poni2_px']):+6.1f})px  loss={float(r['loss']):.1f}")
