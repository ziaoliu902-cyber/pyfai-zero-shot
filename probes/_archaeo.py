# -*- coding: utf-8 -*-
"""考古：mse_v1 / v4 系的成绩单。"""
import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(r"D:\pyfai-project")
for tag in ("mse_v1", "base_v4b", "base_v4c", "base_v4d",
            "base_v3lamp", "px_v2"):
    p = os.path.join("runs", tag, "history.csv")
    if not os.path.isfile(p):
        print(tag, ": no history")
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        print(tag, ": empty")
        continue
    best_i = min(range(len(rows)),
                 key=lambda i: float(rows[i]["val_loss"]))
    r, last = rows[best_i], rows[-1]
    print(f"{tag:12s} {len(rows):2d}ep  best@ep{r['epoch']:>2s} "
          f"val={float(r['val_loss']):8.3f}  dist_best={float(r['mae_dist']):.4f}"
          f"  dist_last={float(last['mae_dist']):.4f}")
