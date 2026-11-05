# -*- coding: utf-8 -*-
"""λ + 曝光双域训练卷：data/train_lamp1（10-28 曝光域差事故的根治数据仓）。

配置 = train_lam_v1 基底 + peak_counts log-uniform (30, 30000)：
  - λ ∈ (0.5, 2.3) Å log 域随机（同 train_lam_v1，aux 第 6 位存 λ）
  - 曝光：最强可见环峰值计数在 log 域 30-30000 均匀（跨 3 个数量级）——
    真实数据实测：ceria 中位 2 光子/px（等效峰值计数 ~几十），AgBh 高动态。
    旧训练仓 (3000,30000) 线性均匀 → 低曝光 regime 完全空白 = NN 真实失准主因。
  - 其余几何/nuisance 与 v3/lam_v1 完全一致——相对 lam_v1 的单变量 = 曝光域。

用法：
  python gen_lamp.py --smoke     # 512 张到 data/_smoke_lamp（验证后手删）
  python gen_lamp.py             # 150k 张到 data/train_lamp1（断点续跑安全）
断点续跑恢复：λ 与曝光配置都从 meta config 恢复（generate_gui 已支持 λ；
peak_counts_loguniform 随 cfg_overrides 全量重放，续跑安全）。
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
SEED = 20261111
LAM_RANGE = (5e-11, 2.3e-10)          # m
COUNT_RANGE = (30.0, 30000.0)         # log-uniform，跨 3 个数量级


def main():
    smoke = "--smoke" in sys.argv
    out = os.path.join(ROOT, "data", "_smoke_lamp" if smoke else "train_lamp1")
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
    print(f"生成 λ+曝光卷：N={total} seed={SEED} "
          f"λ∈{LAM_RANGE[0]*1e10:.2f}-{LAM_RANGE[1]*1e10:.2f}Å "
          f"峰值计数 log-U[{COUNT_RANGE[0]:.0f},{COUNT_RANGE[1]:.0f}] → {out}",
          flush=True)
    run_generation(total, 256, out, SEED, q, threading.Event(),
                   cfg_overrides=dict(wavelength_range=LAM_RANGE,
                                      peak_counts_range=COUNT_RANGE,
                                      peak_counts_loguniform=True))
    stop.set()
    th.join(timeout=5)


if __name__ == "__main__":
    main()
