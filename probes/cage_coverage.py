# -*- coding: utf-8 -*-
"""信任域覆盖率诊断：真值在笼子里吗？（σ 塌缩的直接后果）

背景（10-27 续记二 §sigma_collapse）：σ 头 78~100% 的样本卡在 clamp 下限
0.04979，dist 通道 100% 卡住 → **σ 实际是每通道一个常数** → 信任域
radius = trust_k·σ 退化成**固定盒子**，与样本难度无关。

本探针量化这个结构性天花板：
  对每个样本每个通道，比较 |真值 - NN粗解| 与信任域半径（物理量）：
      覆盖率 = 真值落在盒内的样本比例
  - 覆盖率低 → 精修**物理上不可能**修正该通道 → 配方再调也没用
  - 分档（相/距离）看差距 → 定位"该给谁放宽盒子"

同时验证 §sigma_collapse 的推论「M1cd 的收益其实来自先验收缩而非不确定性建模」：
  比较 NN 粗解与精修后的相关性、以及"精修后误差 ≈ NN 误差×收缩因子"的分布。

用法：python probes/cage_coverage.py
"""
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
from torchfai.phases import PHASE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM_NAMES = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
TRUST_K = 1.0


def main():
    t0 = time.time()
    ds = TorchFAIDataset(os.path.join(ROOT, "data/train_v3"), split="test")
    aux_all = ds._maux[ds._sel]
    actual = (aux_all[:, 1] >= 6) & (aux_all[:, 3] >= 10000) & (aux_all[:, 4] < 0)
    sel = np.where(actual)[0]

    ck = torch.load(os.path.join(ROOT, "runs/base_v3/last.pt"),
                    map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"])
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    meta = json.load(io.open(os.path.join(ROOT, "data/train_v3/meta.json"),
                             encoding="utf-8"))
    cfg = meta["config"]
    dlo, dhi = np.log10(cfg["dist_range"][0]), np.log10(cfg["dist_range"][1])
    from torchfai.refine import DifferentiableRefiner
    ref = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3"))
    Hm, Wm = ref.consts["Hm"], ref.consts["Wm"]
    r_lo, r_hi = ref.consts["r_lo"], ref.consts["r_hi"]
    PIX_MM = ref.db.pixel * 1000.0

    preds, sigs, tps, phases, dphys = [], [], [], [], []
    with torch.no_grad():
        for k in sel:
            g = int(ds._sel[k])
            img = np.asarray(ds._mimgs[g])
            x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].to(dev)
            mu, lv = model(x)
            preds.append(norm_to_param(mu[0].cpu().numpy().astype(np.float64),
                                       ds.norm_consts))
            sigs.append(np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64))
            tps.append(ds._mparams[g].astype(np.float64))
            phases.append(PHASE_NAMES[int(ds._maux[g][0])])
            dphys.append(ds._mparams[g][0])
    P = np.array(preds)
    S = np.array(sigs)
    TP = np.array(tps)
    ph = np.array(phases)
    dphys = np.array(dphys)
    n = len(P)
    print(f"实际工况档 N={n}  {time.time()-t0:.0f}s", flush=True)

    # ---- 信任域半径（物理量）----
    # pn 半径 r → 物理半径
    def phys_radius(sig):
        rad = TRUST_K * sig
        out = np.zeros_like(rad)
        out[:, 0] = 10.0 ** (((rad[:, 0]) / 2.0 * (dhi - dlo))) - 1.0   # 相对误差
        out[:, 1] = rad[:, 1] / 2.0 * Hm                                # px
        out[:, 2] = rad[:, 2] / 2.0 * Wm                                # px
        for j, (lo, hi) in enumerate(zip(r_lo, r_hi)):
            out[:, 3 + j] = rad[:, 3 + j] / 2.0 * (hi - lo)              # mrad
        return out

    RAD = phys_radius(S)
    ERR = np.abs(P - TP)
    # dist 误差转成相对
    rel_dist = np.abs(P[:, 0] - TP[:, 0]) / TP[:, 0]

    out = []
    W = out.append
    W("# 信任域覆盖率诊断（真值在笼子里吗？）\n")
    W(f"- 数据：train_v3 test 实际工况档 N={n}")
    W(f"- 信任域：radius = trust_k·σ（trust_k={TRUST_K}），σ 取 NN 自报")
    W(f"- σ 已证实塌缩（见 sigma_collapse 报告）：dist 100% 卡 clamp 下限\n")

    W("## 1. 总体覆盖率（真值落在信任域盒内的比例）\n")
    W("| 通道 | σ中位 | 信任域半径(物理) | NN误差中位 | **覆盖率** | 覆盖率(严格) |")
    W("|---|---|---|---|---|---|")
    # 单位：dist = 相对距离；poni = 米（探测器物理坐标）；rot = 弧度
    units = ["相对距离", "mm", "mm", "mrad", "mrad", "mrad"]
    scale = [1.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    for j, nm in enumerate(PARAM_NAMES):
        if j == 0:
            e = rel_dist
            r = RAD[:, 0]
        else:
            e = ERR[:, j]
            r = RAD[:, j]
        cov = float((e <= r).mean() * 100)
        strict = float((e < r * 0.8).mean() * 100)
        u = units[j]
        sc = scale[j]
        if j == 0:
            W(f"| {nm} | {np.median(S[:,j]):.5f} | {np.median(r)*100:.2f}% | "
              f"{np.median(e)*100:.2f}% | **{cov:.1f}%** | {strict:.1f}% |")
        else:
            W(f"| {nm} | {np.median(S[:,j]):.5f} | {np.median(r)*sc:.3f} {u} | "
              f"{np.median(e)*sc:.3f} {u} | **{cov:.1f}%** | {strict:.1f}% |")
    W("")

    W("## 2. 分相覆盖率（谁被笼子关在门外）\n")
    W("| 相 | N | " + " | ".join(f"{n_}" for n_ in PARAM_NAMES) + " |")
    W("|---|---|" + "---|" * len(PARAM_NAMES))
    for p_ in sorted(set(ph)):
        m = ph == p_
        cells = []
        for j, nm in enumerate(PARAM_NAMES):
            e = rel_dist if j == 0 else ERR[:, j]
            r = RAD[:, j]
            cells.append(f"{(e[m] <= r[m]).mean()*100:.0f}%")
        W(f"| {p_} | {int(m.sum())} | " + " | ".join(cells) + " |")
    W("")

    W("## 3. 分距离档覆盖率（远距 = 笼子最不够用）\n")
    W("（覆盖率 = 真值落在信任域盒内的样本比例；poni 单位 mm、rot 单位 mrad）\n")
    dq = np.quantile(dphys, [0, .33, .67, 1.0])
    W("| 距离档 (m) | N | " + " | ".join(PARAM_NAMES) + " |")
    W("|---|---|" + "---|" * len(PARAM_NAMES))
    for i in range(3):
        m = (dphys >= dq[i]) & ((dphys <= dq[i+1]) if i == 2 else (dphys < dq[i+1]))
        if m.sum() == 0:
            continue
        cells = []
        for j, nm in enumerate(PARAM_NAMES):
            e = rel_dist if j == 0 else ERR[:, j]
            r = RAD[:, j]
            cells.append(f"{(e[m] <= r[m]).mean()*100:.0f}%")
        W(f"| {dq[i]:.2f}–{dq[i+1]:.2f} | {int(m.sum())} | " + " | ".join(cells) + " |")
    W("")

    W("## 4. 关键推论：M1cd 的收益来自「先验收缩」而非「不确定性建模」\n")
    W("σ·T 里 T<1 的通道（dist 0.317 / poni 0.53）做两件事：")
    W("1. **缩小信任域**（收紧盒子 → 限制搜索）")
    W("2. **放大软先验罚项**（σ 变小 → pen=((pn-pn0)/σ)² 变大 → 拉回 NN 粗解）\n")
    W("第 2 项本质是**向 NN 粗解做收缩（shrinkage）**——"
      "对中位样本是降方差（好），对尾部是锁死错误（坏）。"
      "这正好解释了「中位全线改善、尾部纹丝不动、#321 反而变差 11.78mm」。\n")
    W("验证：用已有 fullpool_dual CSV 算「精修后误差 / NN 误差」的比值分布，"
      "并看精修结果与 NN 粗解的相关性。\n")

    csv_fp = sorted([f for f in os.listdir(os.path.join(ROOT, "probes", "results"))
                      if f.startswith("fullpool_dual_") and f.endswith(".csv")])[-1]
    import csv as _csv
    rows = list(_csv.DictReader(open(os.path.join(ROOT, "probes", "results",
                                                csv_fp),
                                     newline="", encoding="utf-8")))
    m1d = np.array([float(r["m1_dist"]) for r in rows])
    m2d = np.array([float(r["m1cd_dist"]) for r in rows])
    nn_d = ERR[:, 0] * 1000.0     # mm
    W("| 指标 | M1 | M1cd |")
    W("|---|---|---|")
    W(f"| dist 误差中位 (mm) | {np.median(m1d):.4f} | {np.median(m2d):.4f} |")
    W(f"| dist 误差 p90 (mm) | {np.percentile(m1d,90):.4f} | "
      f"{np.percentile(m2d,90):.4f} |")
    W(f"| 与 NN 粗解误差的相关 | {np.corrcoef(m1d, nn_d)[0,1]:.4f} | "
      f"{np.corrcoef(m2d, nn_d)[0,1]:.4f} |")
    W(f"| 误差比 (精修/NN) 中位 | {np.median(m1d/np.maximum(nn_d,1e-9)):.4f} | "
      f"{np.median(m2d/np.maximum(nn_d,1e-9)):.4f} |")
    W("")
    W("→ 若 M1cd 与 NN 粗解误差的相关性**更高**、误差比**更小**，"
      "则证实「M1cd = 更强地收缩到 NN」，而非「更好的不确定性建模」。\n")

    W("## 5. 结论与行动项\n")
    W("1. **σ 塌缩 → 信任域是固定盒子**，与样本难度无关；"
      "覆盖率低的通道/档位是**结构性不可达**，精修配方无法修复。")
    W("2. 应对方向：**按档/按相给信任域设物理半径下限**"
      "（`radius = max(trust_k·σ, r_min[channel, tier])`），"
      "而不是继续用退化的 σ 调 T。")
    W("3. **σ 只应留在软先验里**（作统计权重），不应兼作搜索半径——"
      "这与 refine.py 新增的 `sigma_prior` 解耦方向一致。")
    W("4. σ 头本身要修：clamp 下限 -6.0 卡掉了 78~100% 样本的真实输出，"
      "应放宽下限（如 -12）或改用更稳的不确定性估计（ensemble / MC dropout）。\n")

    ts = time.strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(ROOT, "probes", "results", f"cage_coverage_{ts}.md")
    io.open(fp, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))
    print(f"\n落盘 {fp}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
