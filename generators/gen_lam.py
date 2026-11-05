# -*- coding: utf-8 -*-
"""λ-conditioned 训练卷：data/train_lam_v1（论文 R6 / 对拍 v3 的数据仓）。

配置 = train_v3 基底 + wavelength_range (0.5, 2.3) Å：
  每样本 λ 在 log 域 0.05Å 网格档随机（可行域缓存友好），渲染/标签用 ±0.01Å
  抖动连续值；aux 第 6 位存 λ(Å)。dist/poni/rot/nuisance 采样与 v3 完全一致
  ——单变量扩 λ，便于与 base_v3 对照。

用法：
  python gen_lam.py --smoke     # 512 张到 data/_smoke_lam（验证后手删）
  python gen_lam.py             # 150k 张到 data/train_lam_v1（断点续跑安全）
"""
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_gui import run_generation

ROOT = os.path.dirname(os.path.abspath(__file__))

N_FULL = 150000
SEED = 20261110
LAM_RANGE = (5e-11, 2.3e-10)          # m


def main():
    smoke = "--smoke" in sys.argv
    out = os.path.join(ROOT, "data", "_smoke_lam" if smoke else "train_lam_v1")
    total = 512 if smoke else N_FULL
    os.makedirs(out, exist_ok=True)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def drain():
        t0 = time.time()
        while True:
            try:
                kind, payload = q.get(timeout=2)
            except queue.Empty:
                if stop.is_set():
                    return
                continue
            if kind == "done":
                print(f"DONE status={payload['status']} actual={payload['actual']} "
                      f"rejects={payload['rejects']} {(time.time()-t0)/60:.1f}min",
                      flush=True)
                stop.set()
                return
            if kind == "prog":
                n, tot, el, ph, dist, rej, sh = payload
                rate = n / max(el, 1e-9)
                print(f"{n:6d}/{tot}  {el/60:6.1f}min  {rate:5.1f}/s  "
                      f"ETA {(tot-n)/max(rate,1e-9)/3600:4.1f}h  相={ph:7s} "
                      f"dist={dist:.2f}m  拒绝={rej}  分片={sh}", flush=True)
            elif kind == "log" and ("落盘" in payload or "开始" in payload or "覆盖" in payload):
                print(payload, flush=True)

    th = threading.Thread(target=drain, daemon=True)
    th.start()
    print(f"生成 λ 卷：N={total} seed={SEED} λ∈{LAM_RANGE[0]*1e10:.2f}-{LAM_RANGE[1]*1e10:.2f}Å "
          f"→ {out}", flush=True)
    run_generation(total, 256, out, SEED, q, threading.Event(),
                   cfg_overrides=dict(wavelength_range=LAM_RANGE))
    stop.set()
    th.join(timeout=5)


if __name__ == "__main__":
    main()
