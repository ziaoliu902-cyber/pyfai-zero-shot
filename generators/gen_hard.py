# -*- coding: utf-8 -*-
"""难样本补丁生成（headless，CPU 后台，与 GPU 精修并行）。

【2026-10-26 修正】本文件已改写为 as-run 配方（与 data/hard_patch_v1 两个子目录
的 meta.json 逐项核对）。原方案是"按 refine_v2 卡住画像加权、不补 Si"的 5+1 补丁，
实施时改为"近距全六相均匀补课"：加权方案的相间差异（卡住率 6%-14%）不足以支撑
定向配比，均匀覆盖近距全域更稳；Si 虽卡住率最低(0.6%)但近距段仍属薄弱区，一并补入。
agbh_far 与原方案一致。若按旧版 PATCHES 复现会得到不同目录结构，勿再用。

产出（已由 merge_v3.py 合入 data/train_v3，行 65000..79999，审计 28/28 通过）：
  near_mix : 12000 张，全六相随机，dist 0.10-0.35m，seed 20261026，shard 256
             → data/hard_patch_v1/near_mix（六相各 ~2000）
  agbh_far :  3000 张，AgBh 固定，  dist 0.80-3.00m，seed 20261031，shard 256
             → data/hard_patch_v1/agbh_far

历史靶点画像（refine_v2 全量卡住名单 >5mm，留档备查）：
  近距(0.10-0.35m): Al2O3 13.6% / AgBh 9.9% / Quartz 9.3% / LaB6 8.0% / CeO2 6.0%
  远距(0.80-3.00m): AgBh 6.4%；Si 基本无卡住(0.6%)
"""
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_gui import run_generation
from torchfai.phases import PHASE_NAMES

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "data", "hard_patch_v1")

# (phase, dist_range, total, seed, 输出子目录) —— seed/shard 显式写死，与已落盘 meta 一致
PATCHES = [
    (None,   (0.10, 0.35), 12000, 20261026, "near_mix"),
    ("AgBh", (0.80, 3.00),  3000, 20261031, "agbh_far"),
]


def drain(q, tag):
    """后台线程：把 worker 日志打到控制台。"""
    while True:
        try:
            kind, payload = q.get(timeout=2)
        except queue.Empty:
            continue
        if kind == "done":
            print(f"[{tag}] DONE {payload}", flush=True)
            return
        if kind == "log":
            print(f"[{tag}] {payload}", flush=True)


t_start = time.time()
for i, (ph, dr, total, seed, sub) in enumerate(PATCHES):
    out = os.path.join(BASE, sub)
    print(f"\n===== 补丁 {i+1}/{len(PATCHES)}: {ph or '全相随机'} dist={dr} N={total} "
          f"seed={seed} → {out}", flush=True)
    q = queue.Queue()
    stop = threading.Event()
    th = threading.Thread(target=drain, args=(q, sub), daemon=True)
    th.start()
    run_generation(total, 256, out, seed, q, stop, None, None,
                   phase_idx=(None if ph is None else PHASE_NAMES.index(ph)),
                   dist_range=dr)
    th.join(timeout=10)

print(f"\n全部补丁完成，总耗时 {(time.time()-t_start)/60:.1f} 分钟", flush=True)
