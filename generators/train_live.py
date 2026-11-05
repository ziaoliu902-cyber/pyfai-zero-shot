# -*- coding: utf-8 -*-
"""train.py 的驾驶舱版 CLI：后台线程跑 run_training，主线程实时排水队列。

背景：train.py 的 main() 在 run_training 返回后才排水队列 → 命令行窗口
全程空白（10-29 训练窗口空白事故的根因；GUI 版无此问题）。本文件与其
参数完全一致，替换启动方式即可获得实时输出。Ctrl+C 安全停止。

用法：python -u -X utf8 train_live.py --tag xxx ...（同 train.py 参数）
"""
import argparse
import os
import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import DEFAULT_OPTS, run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_OPTS["data"])
    ap.add_argument("--tag", default=DEFAULT_OPTS["tag"])
    ap.add_argument("--epochs", type=int, default=DEFAULT_OPTS["epochs"])
    ap.add_argument("--batch", type=int, default=DEFAULT_OPTS["batch"])
    ap.add_argument("--lr", type=float, default=DEFAULT_OPTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULT_OPTS["weight_decay"])
    ap.add_argument("--loss", choices=["nll", "mse"], default=DEFAULT_OPTS["loss"])
    ap.add_argument("--warmup-steps", type=int, default=DEFAULT_OPTS["warmup_steps"])
    ap.add_argument("--limit", type=int, default=DEFAULT_OPTS["limit"])
    ap.add_argument("--val-limit", type=int, default=DEFAULT_OPTS["val_limit"])
    ap.add_argument("--seed", type=int, default=DEFAULT_OPTS["seed"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=DEFAULT_OPTS["init_from"])
    ap.add_argument("--lam-cond", action="store_true")
    ap.add_argument("--px-cond", action="store_true")
    ap.add_argument("--poiss-aug", type=float, default=DEFAULT_OPTS["poiss_aug"])
    ap.add_argument("--param-weights", default=DEFAULT_OPTS["param_weights"])
    args = ap.parse_args()

    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(target=run_training, args=(vars(args), q, stop),
                         daemon=True)
    t.start()
    try:
        while t.is_alive():
            try:
                tag, payload = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if tag == "log":
                print(payload, flush=True)
            elif tag == "step":
                ep, s, tot, loss, lr, dt = payload
                print(f"ep{ep} {s:4d}/{tot}  loss={loss:.4f}  lr={lr:.2e}  "
                      f"({dt:.0f}s)", flush=True)
            elif tag == "epoch":
                ep, tr, va, mae, sec = payload
                print(f"== epoch {ep}: train={tr:.4f} val={va:.4f} ({sec:.0f}s) | "
                      + " ".join(f"{n}={v:.4g}" for n, v in zip(
                          ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"),
                          mae)), flush=True)
            elif tag == "done":
                print("DONE:", payload, flush=True)
    except KeyboardInterrupt:
        print("Ctrl+C → 安全停止（批边界检查点）…", flush=True)
        stop.set()
        t.join()
        return
    t.join()


if __name__ == "__main__":
    main()
