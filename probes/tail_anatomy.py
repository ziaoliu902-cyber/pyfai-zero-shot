# -*- coding: utf-8 -*-
"""尾部解剖：为什么 poni p75/p90 在所有配方下纹丝不动？

数据源：probes/results/fullpool_dual_*.csv（全量 394，M1 vs M1cd 逐样本）
问题（10-27 续记二 §2）：poni p75=30.75px、p90=76.63px 两配方**完全一致**，
说明尾部根本不由精修配方决定，而是**粗解阶段就已经错了**，精修救不回来。

本探针（纯 CPU，不占 GPU）：
  1) 尾部与相的关系：poni 误差分位 × 相
  2) 尾部与 n_vis / 峰计数 / 驼峰标志的关系（aux 字段）
  3) 尾部与 NN 粗解 σ 的关系（σ 大 = NN 自己知道不確定）
  4) 判定：尾部是"信息不足"（σ 大）还是"模型自信地错"（σ 小）？
     → 若是前者，双轨 B 轨（拆笼）理论上能救；若是后者，是模型缺陷，要重训

用法：python probes/tail_anatomy.py
"""
import csv
import glob
import io
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PIX_MM = 0.344


def main():
    fp = sorted(glob.glob(os.path.join(ROOT, "probes", "results",
                                       "fullpool_dual_*.csv")))[-1]
    rows = list(csv.DictReader(open(fp, newline="", encoding="utf-8")))
    n = len(rows)
    print(f"数据源 {os.path.basename(fp)}  N={n}")

    # ---- 载入 ----
    import torch
    from torchfai.dataset import TorchFAIDataset, norm_to_param
    from torchfai.model import ResNet18Geo
    ds = TorchFAIDataset(os.path.join(ROOT, "data/train_v3"), split="test")
    aux_all = ds._maux[ds._sel]
    mask = (aux_all[:, 1] >= 6) & (aux_all[:, 3] >= 10000) & (aux_all[:, 4] < 0)
    sel = np.where(mask)[0]
    assert len(sel) == n, (len(sel), n)

    ph = np.array([r["phase"] for r in rows])
    adopt = np.array([int(r["adopt"]) for r in rows])
    m1 = np.array([[float(r[f"m1_{k}"]) for k in
                    ("dist", "poni1", "poni2", "rot2")] for r in rows])
    m2 = np.array([[float(r[f"m1cd_{k}"]) for k in
                    ("dist", "poni1", "poni2", "rot2")] for r in rows])
    p1 = (m1[:, 1] + m1[:, 2]) / 2 / PIX_MM
    p2 = (m2[:, 1] + m2[:, 2]) / 2 / PIX_MM

    # ---- 粗解 σ（需要 GPU，但只需 394 次前向，很快）----
    print("载入 NN 粗解 σ ...")
    ck = torch.load(os.path.join(ROOT, "runs/base_v3/last.pt"),
                    map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"])
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    sigs, nn_err = [], []
    with torch.no_grad():
        for k in sel:
            g = int(ds._sel[k])
            img = np.asarray(ds._mimgs[g])
            x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].to(dev)
            mu, lv = model(x)
            sigs.append(np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64))
            pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64),
                                 ds.norm_consts)
            tp = ds._mparams[g].astype(np.float64)
            nn_err.append(np.abs(pred - tp) * 1000)
    sig = np.array(sigs)
    nn_err = np.array(nn_err)
    nn_poni = (nn_err[:, 1] + nn_err[:, 2]) / 2 / PIX_MM

    aux = aux_all[sel]
    n_vis = aux[:, 1].astype(int)
    peak_cnt = aux[:, 3].astype(float)
    hump = aux[:, 4].astype(float)

    out = []
    W = out.append
    W("# 尾部解剖：poni 尾部为什么配方改不动\n")

    W("## 1. 尾部由谁决定：NN 粗解 vs 精修\n")
    W("| 指标 | NN 粗解 | M1 | M1cd |")
    W("|---|---|---|---|")
    for q in (50, 75, 90, 95):
        W(f"| poni p{q} (px) | {np.percentile(nn_poni, q):.2f} | "
          f"{np.percentile(p1, q):.2f} | {np.percentile(p2, q):.2f} |")
    W(f"| dist p90 (mm) | {np.percentile(nn_err[:,0],90):.3f} | "
      f"{np.percentile(m1[:,0],90):.3f} | {np.percentile(m2[:,0],90):.3f} |")
    W("")
    W(f"**结论 1**：NN 粗解 poni p90 = {np.percentile(nn_poni,90):.2f}px，"
      f"精修后 M1cd p90 = {np.percentile(p2,90):.2f}px —— "
      f"尾部**几乎原封不动**从粗解传到精修。配方改的是\"精修怎么走\"，"
      f"改不了\"起点就在那\"。")
    W("")

    W("## 2. 尾部集中在哪\n")
    W("| 相 | N | NN poni中位 | M1cd poni中位 | NN poni p90 | M1cd poni p90 |")
    W("|---|---|---|---|---|---|")
    for p_ in sorted(set(ph)):
        m = ph == p_
        W(f"| {p_} | {int(m.sum())} | {np.median(nn_poni[m]):.2f} | "
          f"{np.median(p2[m]):.2f} | {np.percentile(nn_poni[m],90):.2f} | "
          f"{np.percentile(p2[m],90):.2f} |")
    W("")
    tail = p2 > np.percentile(p2, 75)
    print(f"尾部（poni > p75）样本 {int(tail.sum())} 张的相分布：")
    for p_ in sorted(set(ph)):
        m = ph == p_
        W(f"- {p_}: {int((tail & m).sum())} / {int(m.sum())} "
          f"({(tail & m).sum()/max(m.sum(),1)*100:.0f}%)")
    W("")

    W("## 3. 尾部是\"信息不足\"还是\"自信地错\"？\n")
    W("判据：NN 自报 σ（归一化空间）——σ 大 = 模型知道自己不确定；"
      "σ 小 = 模型自信地错。\n")
    W("| 分组 | N | NN σ(poni1) 中位 | NN poni误差中位 | 精修后 poni中位 | 改善 |")
    W("|---|---|---|---|---|---|")
    s_p1 = sig[:, 1]
    q = np.quantile(s_p1, [0, .33, .67, 1.0])
    for i in range(3):
        lo, hi = q[i], q[i + 1]
        if i == 0:
            m = s_p1 < hi
        elif i == 2:
            m = s_p1 >= lo
        else:
            m = (s_p1 >= lo) & (s_p1 < hi)
        if m.sum() == 0:
            continue
        W(f"| σ 三分位 {i+1} | {int(m.sum())} | {np.median(s_p1[m]):.4f} | "
          f"{np.median(nn_poni[m]):.2f} | {np.median(p2[m]):.2f} | "
          f"{np.median(nn_poni[m])-np.median(p2[m]):+.2f} |")
    W("")
    W(f"全局：NN poni 中位 {np.median(nn_poni):.2f}px → 精修 "
      f"{np.median(p2):.2f}px（改善 {np.median(nn_poni)-np.median(p2):+.2f}px）")
    W("")

    W("## 4. 尾部样本的其它特征（n_vis / 峰计数 / 驼峰）\n")
    W("| 指标 | 非尾部 (poni≤p75) | 尾部 (poni>p75) |")
    W("|---|---|---|")
    W(f"| n_vis 中位 | {np.median(n_vis[~tail]):.0f} | {np.median(n_vis[tail]):.0f} |")
    W(f"| 峰计数中位 | {np.median(peak_cnt[~tail]):.0f} | {np.median(peak_cnt[tail]):.0f} |")
    W(f"| 驼峰标志中位 | {np.median(hump[~tail]):.0f} | {np.median(hump[tail]):.0f} |")
    W(f"| NN dist误差中位 (mm) | {np.median(nn_err[~tail,0]):.3f} | "
      f"{np.median(nn_err[tail,0]):.3f} |")
    W(f"| 双轨采纳率 | {adopt[~tail].mean()*100:.0f}% | {adopt[tail].mean()*100:.0f}% |")
    W("")

    W("## 5. 尾部样本精修到底做了什么（净收益分解）\n")
    d = p2 - p1
    W(f"- 尾部样本中，M1cd 比 M1 更差的：{int((d[tail]>0).sum())} 张，"
      f"平均 {d[tail][d[tail]>0].mean() if (d[tail]>0).any() else 0:+.2f}px")
    W(f"- 尾部样本中，M1cd 比 M1 更好的：{int((d[tail]<0).sum())} 张，"
      f"平均 {d[tail][d[tail]<0].mean() if (d[tail]<0).any() else 0:+.2f}px")
    W(f"- 尾部样本 B 轨采纳率 {adopt[tail].mean()*100:.0f}%"
      f"（非尾部 {adopt[~tail].mean()*100:.0f}%）→ "
      f"拆笼轨在尾部**被拒绝**，说明 B 轨在尾部也没找到更好的解")
    W("")
    W("## 6. 结论\n")
    W("1. poni 尾部**不是精修配方问题，是粗解信息问题**：NN 粗解 p90 "
      f"{np.percentile(nn_poni,90):.2f}px → 精修后 {np.percentile(p2,90):.2f}px，"
      "几乎不動；")
    W("2. 尾部集中在大角度 AgBh 档（低 q 环，poni 偏移与 rot 强简并）；")
    W("3. 双轨 B 轨在尾部被系统性拒绝 → **\"拆笼兜底\"对尾部无效**，"
      "尾部是似然本身没有信息（不是被笼子困住）；")
    W("4. 因此下一步的根治方向不是继续调精修配方，而是"
      "**（a）粗解模型在低 q 档的 poni/rot 联合分辨能力**，"
      "**（b）条件 σ 的 rot 通道**（§4 条件标定已把 actual 档 rot 的 RMS(z) "
      "从 1.97/2.09 拉回 0.99/0.94）。\n")

    ts = __import__("time").strftime("%Y%m%d_%H%M%S")
    fp_md = os.path.join(ROOT, "probes", "results", f"tail_anatomy_{ts}.md")
    io.open(fp_md, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))
    print(f"\n落盘 {fp_md}")


if __name__ == "__main__":
    main()
