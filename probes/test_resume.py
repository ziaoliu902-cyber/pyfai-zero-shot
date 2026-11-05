# -*- coding: utf-8 -*-
"""断点续跑正确性测试：跑一半强停 → 续跑 → 断言总张数与样本唯一性。

唯一性是关键：RNG 状态恢复正确 ⇒ 续跑样本与已有样本绝不重复。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import time
import queue
import threading
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_gui import run_generation

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "resume_test")
shutil.rmtree(OUT, ignore_errors=True)

# ---- 第一段：目标 40 张，1.2 秒后强停 ----
q1 = queue.Queue()
stop1 = threading.Event()
th1 = threading.Thread(target=run_generation,
                       args=(40, 16, OUT, 123, q1, stop1), daemon=True)
th1.start()
time.sleep(1.2)
stop1.set()
th1.join(timeout=30)
assert not th1.is_alive(), "worker 未按期停止"

with open(os.path.join(OUT, "meta.json"), encoding="utf-8") as f:
    meta1 = json.load(f)
n1 = meta1["actual"]
print(f"第一段停止于 {n1}/40，status={meta1['status']}, rng_state={'有' if meta1.get('rng_state') else '无'}")
assert 0 < n1 < 40, "停止时机不理想（应落在中间）"
assert meta1.get("rng_state"), "停止时未保存 RNG 状态"

# ---- 第二段：续跑（携带 meta1）----
q2 = queue.Queue()
stop2 = threading.Event()
th2 = threading.Thread(target=run_generation,
                       args=(40, 16, OUT, 123, q2, stop2, None, meta1), daemon=True)
th2.start()
th2.join(timeout=120)
assert not th2.is_alive(), "续跑未按期完成"

with open(os.path.join(OUT, "meta.json"), encoding="utf-8") as f:
    meta2 = json.load(f)
print(f"第二段结束：actual={meta2['actual']}, status={meta2['status']}")
assert meta2["status"] == "done" and meta2["actual"] == 40, "续跑未达成目标"

# ---- 全量回读：总张数 & 唯一性 ----
shards = sorted(f for f in os.listdir(OUT) if f.endswith(".npz"))
params_all = np.concatenate([np.load(os.path.join(OUT, s))["params"] for s in shards])
n_unique = np.unique(params_all, axis=0).shape[0]
print(f"回读 {len(shards)} 个分片，共 {len(params_all)} 条，唯一 {n_unique} 条")
assert len(params_all) == 40, "总条数错误"
assert n_unique == 40, f"发现重复样本（唯一 {n_unique}/40）——RNG 状态恢复有误！"

print("PASS：暂停/续跑链路正确（总张数对、样本零重复、状态账目一致）")
