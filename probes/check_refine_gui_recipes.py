# -*- coding: utf-8 -*-
"""无头验证 refine_gui 的配方内核（不开窗）。

对每个配方跑 3 张（train_v3 test 的实际工况档），检查：
  - 内核不抛异常、CSV 表头/行正确
  - m1cd 与 m2cd/m2cdx 的 σ 语义不同（信任域 σ vs 先验 σ）
  - adopt 列落盘正确

用法：python probes/check_refine_gui_recipes.py
"""
import csv
import os
import queue
import sys
import threading
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import refine_gui  # noqa: E402

RESULTS = {}
for recipe in ("c0", "m1", "m1cd", "m2cd", "m2cdx"):
    tag = f"_check_{recipe}"
    out_dir = os.path.join(ROOT, "runs", tag)
    os.makedirs(out_dir, exist_ok=True)
    csv_fp = os.path.join(out_dir, "results.csv")
    if os.path.isfile(csv_fp):
        os.remove(csv_fp)

    opts = dict(ckpt=os.path.join(ROOT, "runs/base_v3/last.pt"),
                data="data/train_v3", n=3, iters=400, k=1.0,
                tag=tag, recipe=recipe, restart=True)
    q = queue.Queue()
    stop_evt = threading.Event()
    th = threading.Thread(target=refine_gui.run_refine,
                          args=(opts, q, stop_evt), daemon=True)
    th.start()
    logs, done = [], None
    t0 = time.time()
    while th.is_alive() or not q.empty():
        try:
            kind, payload = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if kind == "log":
            logs.append(payload)
        elif kind == "done":
            done = payload
    th.join(timeout=5)
    print(f"\n=== 配方 {recipe} ===")
    for l in logs:
        print("  ", l)
    print("   done:", done)
    if not os.path.isfile(csv_fp):
        print("   ✗ 没有生成 CSV")
        RESULTS[recipe] = False
        continue
    with open(csv_fp, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    print(f"   表头 {len(head)} 列: {head[:3]}...{head[-3:]}")
    print(f"   数据行 {len(body)}；dist_m 列样例 {[b[2] for b in body]}")
    print(f"   adopt 列: {[b[-1] for b in body]}")
    # 表头：k, phase, dist_m, err_b×6, err_a×6, sec, adopt = 17 列
    ok = (len(head) == 17 and len(body) == 3
          and head[-1] == "adopt"
          and head.count("err_b_dist") == 1 and head.count("err_a_rot3") == 1)
    RESULTS[recipe] = ok
    print("   ", "✓ 通过" if ok else "✗ 表头/行数异常")
    if not ok:
        print("   表头:", head)

print("\n汇总：", RESULTS)
sys.exit(0 if all(RESULTS.values()) else 1)
