# -*- coding: utf-8 -*-
"""信任域物理半径下限：从数据推导，不拍脑袋。

问题（cage_coverage 报告）：σ 塌缩 → 信任域是固定盒子 →
rot1/rot2 覆盖率仅 44.7%/39.8%（AgBh 档 36%/32%），
**超过一半样本的真值在笼子外面**，精修物理上够不着。

解法：radius = max(trust_k·σ, r_min[channel])，
r_min 从 **val 集的 NN 粗解误差分位数**反推（保证覆盖率 ≥ 目标值）：
    r_min[ch] = quantile(|NN误差_ch| , q) / (scale_ch) / trust_k
用 val 拟合、test 验证——与 σ 标定同一套口径（val 拟合 → test 验证）。

跑三档 q（覆盖率目标 50% / 80% / 95%）+ baseline（无下限），
在 train_v3 test 实际工况档全池 394 上做配对 A/B。

用法：python probes/floor_probe.py [--q 0.5,0.8,0.95] [--n 394]
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM_NAMES = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
TRUST_K = 1.0
SOFT = 0.1
ITERS = 400
ADOPT_NATS = 1.0


@torch.no_grad()
def infer(ckpt_fp, ds):
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    m = ResNet18Geo()
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev).eval()
    from torch.utils.data import DataLoader
    MU, LV, Y, A = [], [], [], []
    for x, y, aux in DataLoader(ds, batch_size=64, shuffle=False, num_workers=0):
        x = x.unsqueeze(1).to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            mu, lv = m(x)
        MU.append(mu.float().cpu().numpy())
        LV.append(lv.float().cpu().numpy())
        Y.append(y.numpy())
        A.append(aux.numpy())
    return (np.concatenate(MU), np.exp(0.5 * np.concatenate(LV)),
            np.concatenate(Y), np.concatenate(A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="0.5,0.8,0.95")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()
    QS = [float(x) for x in a.q.split(",")]
    t0 = time.time()

    ds_val = TorchFAIDataset(os.path.join(ROOT, "data/train_v3"), split="val")
    ds_test = TorchFAIDataset(os.path.join(ROOT, "data/train_v3"), split="test")

    # ---- val 拟合 r_min ----
    mu_v, sig_v, y_v, aux_v = infer(os.path.join(ROOT, "runs/base_v3/last.pt"),
                                    ds_val)
    # 归一化空间的误差：|y - mu|（两者都是归一化参数）
    err_v = np.abs(y_v - mu_v)
    ref = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3"))
    consts = ref.consts
    # 每通道：归一化单位 → 物理单位 的换算尺度（用于反推半径）
    scale = np.array([
        (consts["dhi"] - consts["dlo"]) / 2.0,      # dist: 归一化→log10 m
        consts["Hm"] / 2.0,                         # poni1: → px
        consts["Wm"] / 2.0,                         # poni2: → px
        (consts["r_hi"][0] - consts["r_lo"][0]) / 2.0,
        (consts["r_hi"][1] - consts["r_lo"][1]) / 2.0,
        (consts["r_hi"][2] - consts["r_lo"][2]) / 2.0,
    ])
    floors = {}
    for q in QS:
        # 归一化半径 = 误差分位 / (归一化尺度)  —— 误差本身就是归一化量，
        # 直接用分位数作为归一化半径即可（pn 空间是线性映射）
        floors[q] = np.quantile(err_v, q, axis=0)
    print(f"[val 拟合] N_val={len(err_v)}")
    for q in QS:
        print(f"  q={q:.2f}  r_min(归一化) = "
              f"{np.round(floors[q], 4).tolist()}")
    print(f"  对比 σ 中位 = {np.round(np.median(sig_v,0), 5).tolist()}")
    print(f"  → 即：即使 σ 塌缩，地板比 σ 大 "
          f"{np.round(floors[QS[-1]]/np.maximum(np.median(sig_v,0),1e-9),1).tolist()} 倍\n",
          flush=True)

    # ---- test 池 ----
    aux_t = ds_test._maux[ds_test._sel]
    actual = (aux_t[:, 1] >= 6) & (aux_t[:, 3] >= 10000) & (aux_t[:, 4] < 0)
    sel = np.where(actual)[0]
    if a.n:
        sel = sel[:a.n]
    print(f"[test] 实际工况档 N={len(sel)}", flush=True)

    ck = torch.load(os.path.join(ROOT, "runs/base_v3/last.pt"),
                    map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"])
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    refs, samples = {}, []
    for k in sel:
        g = int(ds_test._sel[k])
        img = np.asarray(ds_test._mimgs[g])
        tp = ds_test._mparams[g].astype(np.float64)
        ph = PHASE_NAMES[int(ds_test._maux[g][0])]
        with torch.no_grad():
            x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].to(dev)
            mu, lv = model(x)
            pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64),
                                 ds_test.norm_consts)
            sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
        if ph not in refs:
            refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v3"))
        samples.append((img, tp, ph, pred, sig))

    # ---- 逐样本跑各配方 ----
    variants = [("base", None)] + [(f"q{q:.2f}", floors[q]) for q in QS]
    res = {v[0]: [] for v in variants}
    t0 = time.time()
    for i, (img, tp, ph, pred, sig) in enumerate(samples):
        r = refs[ph]
        # B 轨共用（无笼无先验，确定性）
        r_b = r.refine(img, pred, iters=ITERS, sigma_norm=None, soft_prior=0.0)
        for name, fl in variants:
            a_ = r.refine(img, pred, iters=ITERS, sigma_norm=sig,
                          trust_k=TRUST_K, soft_prior=SOFT, trust_floor=fl)
            adopt = r_b["pure_loss"] < a_["pure_loss"] - ADOPT_NATS
            p = r_b["params"] if adopt else a_["params"]
            res[name].append(np.abs(p - tp) * 1000)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(samples)}  {el:.0f}s "
                  f"(ETA {el/(i+1)*(len(samples)-i-1):.0f}s)", flush=True)

    E = {k: np.array(v) for k, v in res.items()}
    PIX_MM = ref.db.pixel * 1000.0

    out = []
    W = out.append
    W("# 信任域物理半径下限 A/B（r_min 从 val 误差分位反推）\n")
    W(f"- N={len(samples)}，train_v3 test 实际工况档全池")
    W(f"- 配方：refine_dual(σ 笼 + σ 先验, trust_k={TRUST_K}, soft_prior={SOFT}, "
      f"adopt_nats={ADOPT_NATS})，仅信任域加不加地板不同")
    W(f"- 地板（归一化）：" +
      "；".join(f"q={q:.2f} → {np.round(floors[q],4).tolist()}"
                for q in QS))
    W(f"- σ 中位（归一化）：{np.round(np.median(sig_v,0),5).tolist()}\n")

    W("## 1. 主表\n")
    W("| 配方 | dist MAE | dist 中位 | <0.5mm | >5mm | poni中位px | poni p75 | "
      "poni p90 | rot2中位 | rot2 p90 |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    for name, _ in variants:
        e = E[name]
        pp = (e[:, 1] + e[:, 2]) / 2 / PIX_MM
        W(f"| {name} | {e[:,0].mean():.3f} | {np.median(e[:,0]):.3f} | "
          f"{(e[:,0]<0.5).mean()*100:.1f}% | {int((e[:,0]>5).sum())} | "
          f"{np.median(pp):.2f} | {np.percentile(pp,75):.2f} | "
          f"{np.percentile(pp,90):.2f} | {np.median(e[:,3]):.3f} | "
          f"{np.percentile(e[:,3],90):.3f} |")
    W("")

    W("## 2. 与 base 的配对检验\n")
    W("| 对比 | 通道 | 优 | 劣 | 平 | 符号检验 p | Wilcoxon p |")
    W("|---|---|---|---|---|---|---|")
    for name, _ in variants[1:]:
        for ci, cn in ((0, "dist"), (3, "rot2")):
            d = E[name][:, ci] - E["base"][:, ci]
            w = int((d < 0).sum())
            lo = int((d > 0).sum())
            z = int((d == 0).sum())
            try:
                from scipy.stats import binomtest, wilcoxon
                p = float(binomtest(w, len(d), 0.5).pvalue)
                pw = float(wilcoxon(d).pvalue)
            except Exception:
                p = pw = float("nan")
            W(f"| base → {name} | {cn} | {w} | {lo} | {z} | {p:.3e} | {pw:.3e} |")
    W("")

    W("## 3. 尾部（地板的真正目的）\n")
    W("| 配方 | dist p90 | dist p95 | dist>2mm | poni p75 | poni p90 | rot2 p90 | rot2>20mrad |")
    W("|---|---|---|---|---|---|---|---|")
    for name, _ in variants:
        e = E[name]
        pp = (e[:, 1] + e[:, 2]) / 2 / PIX_MM
        W(f"| {name} | {np.percentile(e[:,0],90):.3f} | "
          f"{np.percentile(e[:,0],95):.3f} | {int((e[:,0]>2).sum())} | "
          f"{np.percentile(pp,75):.2f} | {np.percentile(pp,90):.2f} | "
          f"{np.percentile(e[:,3],90):.3f} | {int((e[:,3]>20).sum())} |")
    W("")

    W("## 4. 误伤检查\n")
    for name, _ in variants[1:]:
        d = E[name][:, 0] - E["base"][:, 0]
        W(f"- {name} vs base：变差>1mm {int((d>1).sum())} 张，"
          f"变差>0.2mm {int((d>0.2).sum())} 张，"
          f"变好>1mm {int((d<-1).sum())} 张，"
          f"均值 {d.mean():+.4f} mm")
    W("")

    ts = time.strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(ROOT, "probes", "results", f"floor_probe_{ts}.md")
    io.open(fp, "w", encoding="utf-8").write("\n".join(out))
    np.savez_compressed(os.path.join(ROOT, "probes", "results",
                                     f"floor_probe_{ts}.npz"),
                        **{k: v for k, v in E.items()},
                        floors=np.array([floors[q] for q in QS]),
                        qs=np.array(QS))
    print("\n".join(out))
    print(f"\n落盘 {fp}")


if __name__ == "__main__":
    main()
