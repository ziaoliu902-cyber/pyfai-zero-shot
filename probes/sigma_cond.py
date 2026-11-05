# -*- coding: utf-8 -*-
"""条件化 σ 标定：把「全局 T」换成「按档分组的 T」。

问题（交接文档 §6）：σ 的失准程度**依赖样本档位**，全局缩放治不了——
    档位          dist   poni1   rot1   rot2
    全体 4800     0.32   0.53   1.37   1.46
    实际工况 378  0.22   1.06   2.72   3.24   ← rot 过度自信 3×
    远距>0.77    0.21   0.63   0.82   0.76
    近距<0.35    0.35   0.47   1.49   1.63
全局 T 对实际工况档的 rot 明显不够（3.24 vs 1.46），对远距又过冲（0.76 vs 1.46）。

方案：**按分群各自拟合 T**，落盘 sigma_T_cond.json（分组规则 + 每组 T），
推理时按样本所属组查表乘 σ。

分组规则（只用推理时可得的信息，不看真值）：
  - 档 A「实际工况」：aux[:,1]≥6 且 aux[:,3]≥10000 且 aux[:,4]<0
  - 档 B「远距」：NN 预测 dist > 0.77（用粗解自身的估计，不用真值 → 可部署）
  - 档 C「近距」：其余
  ⚠ 档 B 用的是【NN 粗解的 dist 预测】而非真值，推理时可算；档 A 的判据来自
    aux（模拟器元数据：可见环数/峰计数/驼峰标志），真实数据上需要 NN 代理，
    本探针同时输出「用 NN 预测替代 aux 判据」的可行性诊断。

流程：
  1) val 集按分组分别拟合 T_g = RMS(z_g)（每通道）
  2) test 集验证：各组标定后 RMS≈1，跨组漂移检查
  3) 输出 T_cond 表 + 与全局 T 的对比表 + 分组样本量

用法：python probes/sigma_cond.py --ckpt runs/base_v3/last.pt --data data/train_v3
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
from torchfai.dataset import TorchFAIDataset, PARAM_NAMES, norm_to_param
from torchfai.model import ResNet18Geo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = ("actual", "far", "near")


@torch.no_grad()
def infer(ckpt_fp, ds, batch=64):
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    m = ResNet18Geo()
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev).eval()
    from torch.utils.data import DataLoader
    MU, LV, Y = [], [], []
    for x, y, aux in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
        x = x.unsqueeze(1).to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            mu, lv = m(x)
        MU.append(mu.float().cpu().numpy())
        LV.append(lv.float().cpu().numpy())
        Y.append(y.numpy())
    return (np.concatenate(MU), np.exp(0.5 * np.concatenate(LV)), np.concatenate(Y))


def zrms(z):
    return np.sqrt((z ** 2).mean(axis=0))


def group_masks(aux, dist_pred):
    """分组：用 aux 判据 + NN 预测 dist（不用真值）。"""
    actual = (aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
    far = (~actual) & (dist_pred > 0.77)
    near = ~actual & ~far
    return {"actual": actual, "far": far, "near": near}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base_v3/last.pt")
    ap.add_argument("--data", default="data/train_v3")
    a = ap.parse_args()
    t0 = time.time()

    ds_val = TorchFAIDataset(os.path.join(ROOT, a.data), split="val")
    ds_test = TorchFAIDataset(os.path.join(ROOT, a.data), split="test")

    mu_v, sig_v, y_v = infer(os.path.join(ROOT, a.ckpt), ds_val)
    aux_v = ds_val._maux[ds_val._sel]
    # 分档用【物理量】 dist：用模型自己训练卷的常数解码 NN 预测（推理时可算）
    dist_pred_v = norm_to_param(mu_v, ds_val.norm_consts)[:, 0]
    print(f"[分档口径] NN 预测物理 dist（m）分位: " +
          " ".join(f"{v:.3f}" for v in np.quantile(dist_pred_v, [0, .25, .5, .75, 1])))
    gm_v = group_masks(aux_v, dist_pred_v)

    z_v = (y_v - mu_v) / sig_v
    T_glob = zrms(z_v)
    print(f"\n[全局 T]（val 拟合，与 sigma_recalib.py 同口径）")
    for j, nm in enumerate(PARAM_NAMES):
        print(f"  {nm:6s} {T_glob[j]:.3f}")

    # ---- 分组拟合 T ----
    T_g = {}
    print(f"\n[分组 T]（val 拟合，RMS(z) 每组每通道）")
    print(f"  {'组':8s}{'N':>6s} " + " ".join(f"{n:>8s}" for n in PARAM_NAMES))
    for g in GROUPS:
        m = gm_v[g]
        if m.sum() < 30:
            print(f"  {g:8s}{int(m.sum()):6d}  （样本太少，跳过）")
            continue
        T_g[g] = zrms(z_v[m])
        print(f"  {g:8s}{int(m.sum()):6d} " +
              " ".join(f"{v:8.3f}" for v in T_g[g]))
    # 全局作为 fallback
    T_g["global"] = T_glob

    # ---- test 验证 ----
    mu_t, sig_t, y_t = infer(os.path.join(ROOT, a.ckpt), ds_test)
    aux_t = ds_test._maux[ds_test._sel]
    dist_pred_t = norm_to_param(mu_t, ds_test.norm_consts)[:, 0]
    gm_t = group_masks(aux_t, dist_pred_t)
    z_t = (y_t - mu_t) / sig_t

    print(f"\n[test 验证] 各组标定后 RMS(z)（目标 ≈1.00）")
    print(f"  {'组':8s}{'N':>6s} {'标定方式':>10s} " +
          " ".join(f"{n:>8s}" for n in PARAM_NAMES))
    verdict = {}
    for g in GROUPS:
        m = gm_t[g]
        if g not in T_g or m.sum() < 30:
            continue
        # 条件标定
        zc = z_t[m] / T_g[g]
        rg = zrms(zc)
        print(f"  {g:8s}{int(m.sum()):6d} {'条件T':>10s} " +
              " ".join(f"{v:8.3f}" for v in rg))
        # 全局 T 对照
        zg = z_t[m] / T_glob
        print(f"  {g:8s}{int(m.sum()):6d} {'全局T':>10s} " +
              " ".join(f"{v:8.3f}" for v in zrms(zg)))
        d = np.abs(rg - 1.0).max()
        dg = np.abs(zrms(zg) - 1.0).max()
        verdict[g] = dict(cond_max_drift=float(d), glob_max_drift=float(dg),
                          N=int(m.sum()))
        print(f"  → 最大漂移：条件 {d:.3f} vs 全局 {dg:.3f}"
              f"（{'条件更好' if d < dg else '全局更好/持平'}）")

    # ---- 分组判据的可部署性诊断：aux 判据 vs NN 预测替代 ----
    # 用 NN 预测 dist 粗略近似 aux 判据的可达性：若 aux 判据不可得，
    # 退化方案是"按 NN 预测 dist 分档"，这里给出该退化方案的样本量
    print(f"\n[可部署性诊断] 若真实数据上拿不到 aux（可见环数/峰计数/驼峰标志）：")
    dq = np.quantile(dist_pred_t, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"  NN 预测 dist（归一化空间）分位: " +
          " ".join(f"{v:.3f}" for v in dq))
    print(f"  → 退化方案：按 NN 预测 dist 四分位分 4 组各拟合 T（无需 aux）")

    # ---- 落盘 ----
    out = dict(
        ckpt=a.ckpt, data=a.data, fit_on="val",
        N_val=int(len(z_v)),
        grouping=dict(
            actual="aux[:,1]>=6 & aux[:,3]>=10000 & aux[:,4]<0",
            far="not actual & NN_pred_dist>0.77",
            near="not actual & not far",
            note="分组只用推理时可得信息；actual 档判据依赖 aux，"
                 "真实数据需 NN 代理或退化到 dist 分档",
        ),
        T_global={n: float(t) for n, t in zip(PARAM_NAMES, T_glob)},
        T_cond={g: {n: float(t) for n, t in zip(PARAM_NAMES, T_g[g])}
                for g in T_g if g in GROUPS},
        verdict=verdict,
        usage="推理后按样本所属组取 T_g，sigma_norm *= T_g[通道]",
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    fp = os.path.join(ROOT, "runs/base_v3/sigma_T_cond.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n落盘 → {fp}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
