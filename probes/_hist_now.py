# -*- coding: utf-8 -*-
"""快读当前训练进度：最近 epoch + best 标记。用法 python probes/_hist_now.py [tag]"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
tag = sys.argv[1] if len(sys.argv) > 1 else "v3far4"

rows = list(csv.DictReader(open(os.path.join(ROOT, "runs", tag,
                                              "history.csv"),
                                encoding="utf-8")))
best = min(float(r["val_loss"]) for r in rows)
best_ep = min(rows, key=lambda r: float(r["val_loss"]))["epoch"]
for r in rows[-4:]:
    star = "  <- best" if r["epoch"] == best_ep else ""
    print(f"ep{int(r['epoch']):2d}: train={float(r['train_loss']):8.3f} "
          f"val={float(r['val_loss']):8.3f}  dist={float(r['mae_dist']):.4f}"
          f"  lr={float(r['lr']):.2e}{star}")
s = json.load(open(os.path.join(ROOT, "runs", tag, "status.json"),
                   encoding="utf-8"))
run = "训练中" if s.get("running") else f"结束({s.get('reason')})"
bv = s.get("best_val")
bs = f"{bv:.3f}" if isinstance(bv, float) else f"史册ep{best_ep}({best:.3f})"
print(f"当前: ep{s.get('epoch')}/{s.get('epochs')} "
      f"step {s.get('step')}/{s.get('steps_per_epoch')}  {run}  "
      f"best_val={bs}")
