# -*- coding: utf-8 -*-
"""Week2 训练内核 + CLI。

内核 run_training(opts, q, stop_evt, pause_evt) 与界面解耦：
q 投递 ("log"/"step"/"epoch"/"done")；stop/pause 在批边界响应（安全）。
train_gui.py 与本文件共用这一内核。

CLI：
  python train.py --tag base_v1 --epochs 50            # 完整训练
  python train.py --tag smoke --epochs 1 --limit 1024  # 冒烟
  python train.py --tag base_v1 --resume               # 断点续训
"""
import sys

# pythonw（GUI 快捷方式）下 stdout/stderr 为 None，直接 reconfigure 会 AttributeError
# → 曾导致训练驾驶舱双击后静默崩溃（2026-10-25 修复）
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import os
import csv
import json
import time
import argparse
import math
import random
import queue

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchfai.dataset import TorchFAIDataset, norm_to_param, PARAM_NAMES
from torchfai.model import ResNet18Geo


DEFAULT_OPTS = dict(data="data/train_v1", tag="base_v1", epochs=50, batch=32,
                    lr=3e-4, weight_decay=0.05, loss="nll", warmup_steps=300,
                    limit=0, val_limit=0, seed=0, resume=False,
                    param_weights=None, init_from=None, lam_cond=False,
                    px_cond=False, poiss_aug=0.0, sig_head=False, sig_weight=1.0)


def batch_lam(aux: torch.Tensor) -> torch.Tensor:
    """从 aux 取每样本波长(Å)：6/7 列卷取第 6 列；5 列旧卷 = 固定 1.5406。"""
    if aux.dim() == 2 and aux.shape[1] >= 6:
        return aux[:, 5].float()
    return torch.full((aux.shape[0],), 1.5406)


def batch_px(aux: torch.Tensor) -> torch.Tensor:
    """从 aux 取每样本像素尺度 k（10-29 视域域扩展）：7 列卷取第 7 列；旧卷 = 1。"""
    if aux.dim() == 2 and aux.shape[1] >= 7:
        return aux[:, 6].float()
    return torch.ones(aux.shape[0])


def batch_sig(aux: torch.Tensor):
    """从 aux 取 σq 归一化标签（v4s1）：8 列卷取第 8 列；无标签卷返回 None。"""
    if aux.dim() == 2 and aux.shape[1] >= 8:
        return aux[:, 7].float()
    return None


def sig_loss(model, aux, dev):
    """σq 辅助损失（模型带 sig_head 且数据带标签时）；否则 0。
    标签 nan（底座行无 σq 真值）→ 掩掉，只对有标签行算 MSE。"""
    if getattr(model, "sig_head", None) is None:
        return None, None
    sig_t = batch_sig(aux)
    if sig_t is None:
        return None, None
    sig_t = sig_t.to(dev, non_blocking=True)
    ok = torch.isfinite(sig_t)
    if not bool(ok.any()):
        return None, None
    p = model.out_sig.view(-1).float()
    return torch.nn.functional.mse_loss(p[ok], sig_t[ok].float()), sig_t


def loss_fn(mu, logvar, y, mode="nll", w=None):
    """返回 (标量损失, 各维 MSE)。NLL = 0.5*((y-mu)²/σ² + logvar) 维求和批求均值。

    w: 可选各维权重 (6,)——物理加权（如放大弱可观测量 rot3 的损失）。
    """
    se = (y - mu) ** 2
    if mode == "mse":
        return se.mean(), se
    nll = 0.5 * (se * torch.exp(-logvar) + logvar)
    if w is not None:
        nll = nll * w
    return nll.sum(dim=1).mean(), se


@torch.no_grad()
def evaluate(model, loader, dev, consts, mode="nll", max_batches=None, w=None,
             lam_cond=False, px_cond=False, sig_w=1.0):
    """返回 (平均损失, 6 维物理 MAE, σq 归一化 MAE 或 None)。"""
    model.eval()
    tot, n = 0.0, 0
    abs_err = np.zeros(6)
    sig_ae, sig_n = 0.0, 0
    for bi, (x, y, aux) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = x.unsqueeze(1).to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        lam = batch_lam(aux).to(dev, non_blocking=True) if lam_cond else None
        px = batch_px(aux).to(dev, non_blocking=True) if px_cond else None
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            mu, logvar = model(x, lam, px)
            l, _ = loss_fn(mu.float(), logvar.float(), y, mode, w)
            l_sig, sig_t = sig_loss(model, aux, dev)
            if l_sig is not None:
                l = l + sig_w * l_sig
        tot += float(l) * len(x); n += len(x)
        if sig_t is not None:
            ok = torch.isfinite(sig_t)
            if bool(ok.any()):
                sig_ae += float((model.out_sig.view(-1).float()
                                 - sig_t).abs()[ok].sum())
                sig_n += int(ok.sum())
        mu_p = norm_to_param(mu.float().cpu().numpy(), consts,
                             batch_px(aux).numpy() if px_cond else None)
        y_p = norm_to_param(y.cpu().numpy(), consts,
                            batch_px(aux).numpy() if px_cond else None)
        abs_err += np.abs(mu_p - y_p).sum(axis=0)
    model.train()
    return tot / max(n, 1), abs_err / max(n, 1), (sig_ae / max(sig_n, 1) if sig_n else None)


def run_training(opts: dict, q: queue.Queue, stop_evt, pause_evt=None):
    """训练内核（后台线程主体）。opts 见 DEFAULT_OPTS。"""
    o = dict(DEFAULT_OPTS); o.update(opts)

    def log(msg):
        q.put(("log", msg))
        try:
            with open(log_fp, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                f.flush()
        except Exception:
            pass

    torch.manual_seed(o["seed"]); np.random.seed(o["seed"]); random.seed(o["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join("runs", o["tag"])
    os.makedirs(out, exist_ok=True)
    last_fp, best_fp = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    hist_fp = os.path.join(out, "history.csv")
    # ── 可观测性：让「任何进程启动的训练」都能被外部实时监视 ──
    # status.json：进度快照（外部监控页/桌面监控器读它）
    # train.log  ：完整日志尾巴（GUI 读它显示实时日志）
    # control    ：外部写 "pause"/"stop"/"resume" 即可远程控制本次训练
    status_fp = os.path.join(out, "status.json")
    log_fp = os.path.join(out, "train.log")
    ctrl_fp = os.path.join(out, "control")
    try:
        if os.path.isfile(ctrl_fp):
            os.remove(ctrl_fp)
    except Exception:
        pass

    def write_status(**kw):
        st = dict(tag=o["tag"], epochs=o["epochs"], batch=o["batch"],
                  lr=o["lr"], loss=o["loss"], data=o["data"],
                  pid=os.getpid(), updated=time.strftime("%H:%M:%S"),
                  running=True)
        st.update(kw)   # 允许调用方覆盖 running（如结束时置 False）
        tmp = status_fp + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False, indent=1)
            os.replace(tmp, status_fp)      # 原子替换，避免读到半个文件
        except Exception:
            pass

    def read_ctrl():
        try:
            with open(ctrl_fp, encoding="utf-8") as f:
                return f.read().strip().lower()
        except Exception:
            return ""

    tr = TorchFAIDataset(o["data"], "train")
    va = TorchFAIDataset(o["data"], "val")
    consts = tr.norm_consts
    tl = DataLoader(tr, batch_size=o["batch"], shuffle=True, num_workers=0,
                    pin_memory=(dev.type == "cuda"), drop_last=True)
    vl = DataLoader(va, batch_size=o["batch"], shuffle=False, num_workers=0)
    log(f"train={len(tr)} val={len(va)} dev={dev} loss={o['loss']} tag={o['tag']}")
    if o.get("lam_cond") and tr._maux is not None and tr._maux.shape[1] < 6:
        log("⚠ lam_cond=True 但数据卷 aux 无 λ 列（旧 5 列卷）→ λ 将恒为 1.5406，"
            "λ 触须学不到东西！请配 train_lam_v1（aux 6 列）。")
    write_status(running=True, epoch=0, step=0, steps_per_epoch=0, loss=None)

    model = ResNet18Geo(lam_cond=bool(o.get("lam_cond")),
                        px_cond=bool(o.get("px_cond")),
                        sig_head=bool(o.get("sig_head"))).to(dev)
    log(f"ResNet18Geo params {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
        + (" [λ-conditioned]" if o.get("lam_cond") else "")
        + (" [px-conditioned]" if o.get("px_cond") else "")
        + (" [σq-head]" if o.get("sig_head") else ""))
    if o.get("init_from"):
        # 微调起步：只加载模型权重（fresh 优化器/调度器/epoch 计数）。
        # 与 --resume 的区别：resume 恢复完整训练状态并续编号到同一 runs 目录；
        # init_from 是"新实验、旧知识"，适合 v4-lite 这类补课微调。
        # lam_cond/px_cond/sig_head 模型从旧权重热启动：strict=False，
        # 新增触须/头保持零初始化（初始输出与源模型逐位一致——已单测验证）。
        ck = torch.load(o["init_from"], map_location=dev, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd, strict=not (o.get("lam_cond") or o.get("px_cond")
                                              or o.get("sig_head")))
        log(f"init_from：已加载 {o['init_from']} 的模型权重"
            f"（best_val={ck.get('best_val') if isinstance(ck, dict) else '?'}）")
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"],
                            weight_decay=o["weight_decay"])
    steps_per_epoch = (min(len(tr), o["limit"]) if o["limit"] else len(tr)) // o["batch"]
    total_steps = max(o["epochs"] * steps_per_epoch, 1)
    warmup = o["warmup_steps"]

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    # 物理加权（如 "1,1,1,1,1,5" 放大 rot3）——None 表示全 1
    if o.get("param_weights"):
        w_t = torch.tensor([float(v) for v in str(o["param_weights"]).split(",")],
                           dtype=torch.float32, device=dev)
        assert w_t.shape == (6,), "param_weights 必须是 6 个逗号分隔数"
        log(f"param_weights = {w_t.tolist()}")
    else:
        w_t = None

    start_ep, best_val, hist = 0, float("inf"), []
    if o["resume"] and os.path.isfile(last_fp):
        ck = torch.load(last_fp, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, best_val, hist = ck["epoch"] + 1, ck["best_val"], ck["hist"]
        log(f"续训：从 epoch {start_ep} 起（best_val={best_val:.4f}）")
    elif os.path.isfile(hist_fp):
        os.remove(hist_fp)          # 全新训练覆盖旧 history，防混淆

    if not os.path.isfile(hist_fp):
        with open(hist_fp, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr",
                                    *["mae_" + n for n in PARAM_NAMES], "sec"])

    model.train()
    ep = start_ep
    stop_reason = "done"
    while ep < o["epochs"]:
        t0 = time.time()
        run, rn = 0.0, 0
        step = ep * steps_per_epoch
        for bi, (x, y, aux) in enumerate(tl):
            if o["limit"] and (bi + 1) * o["batch"] > o["limit"]:
                break
            if stop_evt.is_set():
                stop_reason = "stopped"
                break
            while pause_evt is not None and pause_evt.is_set() and not stop_evt.is_set():
                time.sleep(0.15)
            if stop_evt.is_set():
                stop_reason = "stopped"
                break
            step += 1
            x = x.unsqueeze(1).to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            # 泊松降曝增强：x=log1p(img)。回线性域按 α 降曝重抽再回 log 域。
            # α 每样本 log-uniform [poiss_aug, 1]——教网络所有曝光档（真实短曝光
            # 数据中位 2 光子/px vs 训练 ~350，10-28 域差事故的修复）。val 不增强。
            if o.get("poiss_aug"):
                amin = float(o["poiss_aug"])
                a = torch.exp(torch.rand(x.shape[0], 1, 1, 1, device=dev,
                                         dtype=torch.float32) * math.log(amin))
                lin = torch.expm1(x.float())
                x = torch.log1p(torch.poisson(lin * a) / a).to(x.dtype)
            lam = batch_lam(aux).to(dev, non_blocking=True) if o.get("lam_cond") else None
            px = batch_px(aux).to(dev, non_blocking=True) if o.get("px_cond") else None
            mode = "mse" if (o["loss"] == "nll" and step <= warmup) else o["loss"]
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                mu, logvar = model(x, lam, px)
                l, _ = loss_fn(mu, logvar, y, mode, w_t)
                l_sig, _ = sig_loss(model, aux, dev)
                if l_sig is not None:
                    l = l + o["sig_weight"] * l_sig
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += float(l.detach()) * len(x); rn += len(x)
            # 外部控制（monitor_gui 写 control 文件）
            c = read_ctrl()
            if c == "stop":
                stop_reason = "stopped"
                log("收到外部 stop 指令 → 停止")
                break
            if c == "pause":
                # 外部挂起：直接阻塞在此，不动 pause_evt（那是 GUI 自己的暂停通道，
                # 动它会覆盖用户在 GUI 上的暂停状态）
                log("收到外部 pause 指令 → 挂起（等 resume）")
                while read_ctrl() == "pause" and not stop_evt.is_set():
                    time.sleep(0.2)
                log("收到外部 resume 指令 → 继续")
            if (bi + 1) % 25 == 0:
                q.put(("step", (ep, bi + 1, steps_per_epoch, run / max(rn, 1),
                                sched.get_last_lr()[0], time.time() - t0)))
                write_status(epoch=ep, step=bi + 1, steps_per_epoch=steps_per_epoch,
                             loss=run / max(rn, 1), lr=sched.get_last_lr()[0],
                             epoch_sec=round(time.time() - t0, 1))
        if stop_reason == "stopped":
            # 紧急检查点：epoch 记为当前值 → resume 将重跑本 epoch（幂等，无损）
            ck = dict(model=model.state_dict(), opt=opt.state_dict(),
                      sched=sched.state_dict(), scaler=scaler.state_dict(),
                      epoch=ep, best_val=best_val, hist=hist, opts=o)
            torch.save(ck, last_fp)
            log(f"epoch {ep} 中途停止 → 已存紧急检查点 last.pt（重跑将从 epoch {ep} 起）")
            break

        val_loss, mae, sig_mae = evaluate(model, vl, dev, consts, o["loss"],
                                          max_batches=o["val_limit"] or None, w=w_t,
                                          lam_cond=bool(o.get("lam_cond")),
                                          px_cond=bool(o.get("px_cond")),
                                          sig_w=o["sig_weight"])
        sec = time.time() - t0
        log(f"epoch {ep}: train={run/max(rn,1):.4f} val={val_loss:.4f} ({sec:.0f}s) | "
            + " ".join(f"{n}={v:.4g}" for n, v in zip(PARAM_NAMES, mae))
            + (f" sigq={sig_mae:.4f}" if sig_mae is not None else ""))
        with open(hist_fp, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, round(run/max(rn,1), 5), round(val_loss, 5),
                                    f"{sched.get_last_lr()[0]:.2e}",
                                    *map(lambda v: f"{v:.5g}", mae), round(sec, 1)])
        hist.append(dict(ep=ep, train=run/max(rn,1), val=val_loss, mae=mae.tolist()))
        ck = dict(model=model.state_dict(), opt=opt.state_dict(),
                  sched=sched.state_dict(), scaler=scaler.state_dict(),
                  epoch=ep, best_val=best_val, hist=hist, opts=o)
        torch.save(ck, last_fp)
        if val_loss < best_val:
            best_val = val_loss
            ck["best_val"] = best_val
            torch.save(ck, best_fp)
            log(f"   新最优 val={best_val:.4f} → best.pt")
        q.put(("epoch", (ep, run / max(rn, 1), val_loss, mae.tolist(), sec)))
        write_status(epoch=ep, step=steps_per_epoch, steps_per_epoch=steps_per_epoch,
                     loss=run / max(rn, 1), val_loss=val_loss, lr=sched.get_last_lr()[0],
                     mae={n: float(v) for n, v in zip(PARAM_NAMES, mae)},
                     best_val=best_val, epoch_sec=round(sec, 1))
        ep += 1

    if dev.type == "cuda":
        log(f"峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    write_status(running=False, finished=time.strftime("%H:%M:%S"),
                 reason=stop_reason, last_ep=ep - 1, best_val=best_val)
    q.put(("done", dict(reason=stop_reason, last_ep=ep - 1, best_val=best_val,
                        tag=o["tag"])))


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
    ap.add_argument("--init-from", default=DEFAULT_OPTS["init_from"],
                    help="热启动权重（如 runs/base_v3/last.pt）：新实验、旧知识")
    ap.add_argument("--lam-cond", action="store_true",
                    help="λ-conditioned 模型（数据卷 aux 6 列；配 train_lam_v1）")
    ap.add_argument("--px-cond", action="store_true",
                    help="px-conditioned 模型（数据卷 aux 7 列；配 train_px1）")
    ap.add_argument("--sig-head", action="store_true",
                    help="σq 辅助头（v4s1 多任务解耦；数据卷需 sigq.npy）")
    ap.add_argument("--sig-weight", type=float, default=DEFAULT_OPTS["sig_weight"],
                    help="σq 辅助损失权重（默认 1.0）")
    ap.add_argument("--poiss-aug", type=float, default=DEFAULT_OPTS["poiss_aug"],
                    help="泊松降曝增强下限 α（如 0.002 = 每样本随机降到 0.2%%-100%% 曝光；0=关）")
    ap.add_argument("--param-weights", default=DEFAULT_OPTS["param_weights"],
                    help="6 个逗号分隔的损失权重，如 1,1,1,1,1,5 放大 rot3")
    args = ap.parse_args()
    opts = vars(args)

    q = queue.Queue()
    run_training(opts, q, threading.Event())
    while not q.empty():
        tag, payload = q.get_nowait()
        if tag == "step":
            ep, s, tot, loss, lr, dt = payload
            print(f"ep{ep} {s}/{tot} loss={loss:.4f} lr={lr:.2e}")
        elif tag == "done":
            print("done:", payload)


if __name__ == "__main__":
    import threading
    main()
