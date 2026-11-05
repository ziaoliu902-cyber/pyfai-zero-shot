# -*- coding: utf-8 -*-
"""考古 v2：v4 系 vs base_v3 的数据卷/配置对比（status.json + history 首行）。"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(r"D:\pyfai-project")
for tag in ("base_v3", "base_v3lam", "base_v3lamp", "base_v4b",
            "base_v4c", "base_v4d", "px_v1", "mse_v1"):
    d = os.path.join("runs", tag)
    s = {}
    try:
        s = json.load(open(os.path.join(d, "status.json"),
                           encoding="utf-8"))
    except Exception:
        pass
    rows = []
    try:
        rows = list(csv.DictReader(open(os.path.join(d, "history.csv"),
                                        encoding="utf-8")))
    except Exception:
        pass
    first = f"ep0 train={float(rows[0]['train_loss']):.2f}" if rows else "-"
    last = (f"末ep train={float(rows[-1]['train_loss']):.2f} "
            f"dist={float(rows[-1]['mae_dist']):.4f}") if rows else "-"
    print(f"{tag:12s} data={s.get('data','?'):22s} ep={s.get('epochs','?'):>3} "
          f"init={s.get('init_from', s.get('pid','?'))}")
    print(f"             {first}  {last}")
