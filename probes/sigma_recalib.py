# -*- coding: utf-8 -*-
"""σ 温度标定：σ' = T·σ（每通道单参数，NLL 解析最优 T = RMS(z)）。

背景（10-26/27 交接）：σ 头过度保守，zRMS 0.26-0.37（理想 1.0）、|err|/σ 中位
0.09-0.16（理想 0.674）→ σ 高估 ~4-6×；信任域偏松、软先验偏弱。

流程：
  1) val 集（4800，参与过选型但未训练）拟合 T_j = RMS(z_j)
  2) 诊断：T 是否随 档（全体/实际工况/远距）/ σ 五分位 稳定
     —— 若各子群 T 相近 → 全局缩放够；若差大 → 需条件标定（记录，不在本次做）
  3) test 集验证：z' = z/T 后 RMS≈1、覆盖 68/95 对齐标称
  4) 落盘 runs/<tag>/sigma_T.json（含 fit_on/diag，refine 调用方乘上即可）

用法：python probes/sigma_recalib.py --ckpt runs/base_v3/last.pt --data data/train_v3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, PARAM_NAMES
from torchfai.model import ResNet18Geo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def infer(ckpt_fp, ds, batch=64):
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    lam_cond = any("lam_proj" in k for k in sd)      # λ 模型自动检测
    m = ResNet18Geo(lam_cond=lam_cond)
    m.load_state_dict(sd)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev).eval()
    from torch.utils.data import DataLoader
    MU, LV, Y = [], [], []
    for x, y, aux in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
        x = x.unsqueeze(1).to(dev, non_blocking=True)
        lam = aux[:, 5].to(dev) if (lam_cond and aux.dim() == 2 and aux.shape[1] >= 6) else None
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            mu, lv = m(x, lam)
        MU.append(mu.float().cpu().numpy())
        LV.append(lv.float().cpu().numpy())
        Y.append(y.numpy())
    return (np.concatenate(MU), np.exp(0.5 * np.concatenate(LV)), np.concatenate(Y))


def zrms(z):
    return np.sqrt((z ** 2).mean(axis=0))


def cov(z, k):
    return (np.abs(z) < k).mean(axis=0) * 100


def report(tag, z, extra=""):
    r = zrms(z)
    c68, c95 = cov(z, 1.0), cov(z, 1.96)
    med = np.median(np.abs(z), axis=0)
    print(f"\n[{tag}] {extra}")
    print(f"  {'通道':6s} {'RMS(z)':>7s} {'|z|中位':>8s} {'68%':>6s} {'95%':>6s}")
    for j, nm in enumerate(PARAM_NAMES):
        print(f"  {nm:6s} {r[j]:7.3f} {med[j]:8.3f} {c68[j]:5.0f}% {c95[j]:5.0f}%")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base_v3/last.pt")
    ap.add_argument("--data", default="data/train_v3")
    a = ap.parse_args()
    t0 = time.time()

    ds_val = TorchFAIDataset(os.path.join(ROOT, a.data), split="val")
    ds_test = TorchFAIDataset(os.path.join(ROOT, a.data), split="test")
    mu, sig, y = infer(os.path.join(ROOT, a.ckpt), ds_val)
    z_val = (y - mu) / sig
    aux_v = ds_val._maux[ds_val._sel]

    # ---- 1) 拟合 T ----
    T = zrms(z_val)
    report("val 原始", z_val, "（过度保守：RMS<1）")

    # ---- 2) 条件稳定性诊断 ----
    dist_v = ds_val._mparams[ds_val._sel][:, 0]
    nv = aux_v[:, 1].astype(int)
    tiers = {
        "全体": np.ones(len(z_val), bool),
        "实际工况": (nv >= 6) & (aux_v[:, 3] >= 10000) & (aux_v[:, 4] < 0),
        "远距>0.77": dist_v > 0.77,
        "近距<0.35": dist_v < 0.35,
    }
    print("\n[条件诊断] 各子群 RMS(z)（看全局 T 是否够用；子群值应接近全局）")
    print(f"  {'档':10s} {'N':>5s} " + " ".join(f"{n:>8s}" for n in PARAM_NAMES))
    for k, m in tiers.items():
        print(f"  {k:10s} {int(m.sum()):5d} " +
              " ".join(f"{v:8.3f}" for v in zrms(z_val[m])))
    print("\n[σ 五分位诊断] dist 通道：按预测 σ 分箱的 RMS(z)（恒定≈全局→缩放即可）")
    sj = np.argsort(sig[:, 0])
    for q in range(5):
        b = sj[q * len(sj) // 5:(q + 1) * len(sj) // 5]
        r = float(np.sqrt((z_val[b, 0] ** 2).mean()))
        print(f"  σ五分位{q+1}: RMS(z_dist)={r:.3f} (σ均值 {sig[b,0].mean():.4f})")

    # ---- 3) test 验证 ----
    mu_t, sig_t, y_t = infer(os.path.join(ROOT, a.ckpt), ds_test)
    z_test = (y_t - mu_t) / sig_t
    z_corr = z_test / T                     # 标定后
    report("test 原始", z_test)
    r_corr = report("test 标定后(z/T)", z_corr, f"T={['%.3f' % t for t in T]}")
    drift = np.abs(r_corr - 1.0)
    print(f"\n  标定后 |RMS-1| 最大通道漂移 {drift.max():.3f}"
          f"（<0.1 = val 拟合的 T 在 test 上同样成立）")

    # ---- 4) 落盘 ----
    out = dict(ckpt=a.ckpt, data=a.data, fit_on="val", N_val=int(len(z_val)),
               T={n: float(t) for n, t in zip(PARAM_NAMES, T)},
               rms_before={n: float(v) for n, v in zip(PARAM_NAMES, zrms(z_test))},
               rms_after={n: float(v) for n, v in zip(PARAM_NAMES, r_corr)},
               usage="推理后 sigma_norm *= T[通道] 再喂 refine（信任域/软先验即时生效）",
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    fp = os.path.join(ROOT, os.path.dirname(a.ckpt), "sigma_T.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n落盘 → {fp}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
