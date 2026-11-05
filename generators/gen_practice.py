# -*- coding: utf-8 -*-
"""v4-lite 微调数据补丁：实际工况段 + 常规混合段（两路并行）。

practice : 12000 张，实际工况画像 —— min_visible_peaks=6（多环）、
           hump_prob=0.05（几乎无驼峰）、peak_counts 1万-4万（清晰亮环）、
           dist 0.10-1.00m（多环几何可行区），seed 20261101
           → data/practice_patch_v1/practice
regular  :  8000 张，默认配置全距离段（防遗忘的"老菜"），seed 20261102
           → data/practice_patch_v1/regular

依据：交接文档 §8/§9 —— 实际工况画像在训练集仅 8%，base_v3 在该档
粗解 poni 5.15px 三档最差；病根在分布失配，用补课段微调治。
产出后续由 merge_ft.py 合成 data/finetune_v1（20k）。
进度可见：控制台日志 + probes/cockpit.py 网页驾驶舱（生成进度卡）。
"""
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_gui import run_generation

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "data", "practice_patch_v1")

# (子目录, total, seed, dist_range, cfg_overrides)
PATCHES = [
    ("practice", 12000, 20261101, (0.10, 1.00),
     dict(min_visible_peaks=6, hump_prob=0.05, peak_counts_range=(10000.0, 40000.0))),
    ("regular", 8000, 20261102, (0.10, 3.00), None),
]

PRAC = {"practice": "实际工况段（多环/亮/无驼峰）", "regular": "常规防遗忘段"}


def worker(sub, total, seed, dr, ov, done):
    q = queue.Queue()

    def drain():
        t0 = time.time()
        while True:
            try:
                kind, payload = q.get(timeout=2)
            except queue.Empty:
                continue
            if kind == "done":
                print(f"[{sub}] DONE status={payload['status']} "
                      f"actual={payload['actual']} rejects={payload['rejects']} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
                done[sub] = True
                return
            if kind == "prog":
                n, tot, el, ph, dist, rej, sh = payload
                print(f"[{sub}] {n:5d}/{tot}  {el/60:5.1f}min  相={ph:6s} "
                      f"dist={dist:.2f}m  拒绝={rej}  分片={sh}", flush=True)
            elif kind == "log" and ("落盘" in payload or "覆盖" in payload or "开始" in payload):
                print(f"[{sub}] {payload}", flush=True)

    threading.Thread(target=drain, daemon=True).start()
    run_generation(total, 256, os.path.join(BASE, sub), seed, q,
                   threading.Event(), cfg_overrides=ov, dist_range=dr)


if __name__ == "__main__":
    os.makedirs(BASE, exist_ok=True)
    done = {}
    ths = []
    for sub, total, seed, dr, ov in PATCHES:
        print(f"启动 {sub}：{PRAC[sub]} N={total} dist={dr} ov={ov}", flush=True)
        t = threading.Thread(target=worker, args=(sub, total, seed, dr, ov, done))
        t.start(); ths.append(t); time.sleep(1)
    for t in ths:
        t.join()
    ok = len(done) == len(PATCHES)
    print(f"\n{'全部完成' if ok else '有失败！'}：{done}", flush=True)
    sys.exit(0 if ok else 1)
