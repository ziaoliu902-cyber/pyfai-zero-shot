# -*- coding: utf-8 -*-
"""模型评估：六参数误差统计 + 分桶短板分析 + 不确定度标定 + rot3 解剖。

用法：
    python eval.py                                    # 默认 best.pt / test 集
    python eval.py --ckpt runs/base_v1/last.pt --split val

说明：
    - val 集参与过 best 模型选择，最终成绩单用 test 集（从未参与训练/选型）
    - 标定在归一化空间做：z=(y-μ)/σ 应近似 N(0,1)；coverage68/95 与 1 对齐即诚实
    - 输出：runs/<tag>/eval_report.md + eval_plots.png + eval_raw.npz
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import os
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchfai.dataset import TorchFAIDataset, norm_to_param, PARAM_NAMES
from torchfai.model import ResNet18Geo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNITS = ("mm", "mm", "mm", "mrad", "mrad", "mrad")
SCALE = (1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0)   # m→mm, rad→mrad


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


@torch.no_grad()
def infer(ckpt_fp, data, split, batch=64):
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    ds = TorchFAIDataset(data, split)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    MU, LV, Y, AUX = [], [], [], []
    for x, y, aux in loader:
        x = x.unsqueeze(1).to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            mu, logvar = model(x)
        MU.append(mu.float().cpu().numpy())
        LV.append(logvar.float().cpu().numpy())
        Y.append(y.numpy())
        AUX.append(aux.numpy())
    mu_n = np.concatenate(MU); lv_n = np.concatenate(LV)
    y_n = np.concatenate(Y); aux = np.concatenate(AUX)
    sig_n = np.exp(0.5 * lv_n)
    consts = ds.norm_consts
    mu_p = norm_to_param(mu_n, consts)
    y_p = norm_to_param(y_n, consts)
    return dict(mu_n=mu_n, sig_n=sig_n, y_n=y_n, mu_p=mu_p, y_p=y_p, aux=aux,
                consts=consts, epoch=ck.get("epoch"), best_val=ck.get("best_val"),
                phase_names=ds.phase_names or [f"P{i}" for i in range(6)], N=len(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base_v1/best.pt")
    ap.add_argument("--data", default="data/train_v1")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    R = infer(a.ckpt, a.data, a.split)
    err_p = (R["mu_p"] - R["y_p"]) * SCALE                 # 物理单位误差
    abs_err = np.abs(err_p)
    z = (R["y_n"] - R["mu_n"]) / R["sig_n"]                # 标定用
    tag = os.path.basename(os.path.dirname(a.ckpt))
    outdir = os.path.dirname(a.ckpt)

    L = []                                                  # report lines
    L.append(f"# 评估报告 — {tag} ({os.path.basename(a.ckpt)})\n")
    L.append(f"- 时间：{datetime.datetime.now():%Y-%m-%d %H:%M}  ")
    L.append(f"- 检查点：epoch {R['epoch']}，选型 val_loss={R['best_val']:.3f}")
    L.append(f"- 评估集：**{a.split}**（N={R['N']}，从未参与训练"
             f"{'/选型' if a.split == 'test' else ''}）\n")

    # ---------------- 1. 总体统计 ----------------
    L.append("## 1. 总体成绩\n")
    rows = []
    for j, nm in enumerate(PARAM_NAMES):
        e = abs_err[:, j]
        rows.append([nm, f"{e.mean():.2f}", f"{np.median(e):.2f}",
                     f"{np.sqrt((e**2).mean()):.2f}", f"{np.percentile(e, 95):.2f}",
                     UNITS[j]])
    L.append(md_table(["参数", "MAE", "中位数", "RMSE", "P95", "单位"], rows))
    L.append("")

    # ---------------- 2. 坍缩检查（学会 vs 躺平）----------------
    L.append("## 2. 坍缩检查（预测分布 vs 真值分布）\n")
    rows = []
    for j, nm in enumerate(PARAM_NAMES):
        t, p = R["y_p"][:, j], R["mu_p"][:, j]
        corr = np.corrcoef(t, p)[0, 1]
        base = np.abs(t - t.mean()).mean() * SCALE[j]   # 猜均值基线（同单位）
        rows.append([nm, f"{t.std():.4f}", f"{p.std():.4f}", f"{corr:+.3f}",
                     f"{abs_err[:, j].mean():.3f}", f"{base:.3f}",
                     "学会" if abs_err[:, j].mean() < 0.7 * base else "≈躺平"])
    L.append(md_table(["参数", "真值σ", "预测σ", "相关r", "MAE",
                       "猜均值MAE", "判定"], rows))
    L.append("")

    # ---------------- 3. 分桶短板 ----------------
    L.append("## 3. 分桶分析\n")
    dist_t = R["y_p"][:, 0]
    edges = np.round(10 ** np.linspace(np.log10(0.1), np.log10(3.0), 6), 3)
    bi = np.clip(np.digitize(dist_t, edges) - 1, 0, len(edges) - 2)

    # dist 桶
    L.append("### 3.1 按探测器距离分桶\n")
    rows = []
    for k in range(len(edges) - 1):
        m = bi == k
        if m.sum() == 0:
            continue
        rows.append([f"{edges[k]}–{edges[k+1]}m", int(m.sum())] +
                    [f"{abs_err[m, j].mean():.2f}" for j in range(6)])
    L.append(md_table(["距离区间", "N", *[f"{n}({u})" for n, u in
                                         zip(PARAM_NAMES, UNITS)]], rows))
    L.append("")

    # phase 桶
    L.append("### 3.2 按物相分桶\n")
    rows = []
    for k, nm in enumerate(R["phase_names"]):
        m = R["aux"][:, 0].astype(int) == k
        rows.append([nm, int(m.sum())] +
                    [f"{abs_err[m, j].mean():.2f}" for j in range(6)])
    L.append(md_table(["物相", "N", *[f"{n}({u})" for n, u in
                                     zip(PARAM_NAMES, UNITS)]], rows))
    L.append("")

    # n_vis 桶
    L.append("### 3.3 按可见环数量分桶\n")
    nv = R["aux"][:, 1].astype(int)
    rows = []
    for k, lab in [(2, "2"), (3, "3"), (4, "4"), (99, "≥5")]:
        m = (nv >= 5) if k == 99 else (nv == k)
        if m.sum() == 0:
            continue
        rows.append([lab, int(m.sum())] +
                    [f"{abs_err[m, j].mean():.2f}" for j in range(6)])
    L.append(md_table(["可见环数", "N", *[f"{n}({u})" for n, u in
                                         zip(PARAM_NAMES, UNITS)]], rows))
    L.append("")

    # ---------------- 4. 不确定度标定 ----------------
    L.append("## 4. 不确定度标定（z=(y-μ)/σ 应≈N(0,1)）\n")
    rows = []
    for j, nm in enumerate(PARAM_NAMES):
        zz = z[:, j]
        rows.append([nm, f"{R['sig_n'][:, j].mean():.3f}",
                     f"{np.sqrt((zz**2).mean()):.3f}",
                     f"{(np.abs(zz) < 1).mean()*100:.0f}%",
                     f"{(np.abs(zz) < 1.96).mean()*100:.0f}%"])
    L.append(md_table(["参数", "平均σ(归一化)", "RMS(z)", "覆盖68%", "覆盖95%"], rows))
    L.append("\n理想值：RMS(z)=1，覆盖 68%/95%。RMS(z)>1 = 过度自信，<1 = 过度保守。\n")

    # ---------------- 5. rot3 解剖 ----------------
    L.append("## 5. rot3 专题解剖\n")
    t3, p3 = R["y_p"][:, 5], R["mu_p"][:, 5]
    corr3 = np.corrcoef(t3, p3)[0, 1]
    s3 = R["sig_n"][:, 5].mean()
    phys_half = (R["consts"]["r_hi"][2] - R["consts"]["r_lo"][2]) / 2
    L.append(f"- 真值范围 ±{phys_half:.2f} rad；真值σ={t3.std():.4f}，"
             f"预测σ={p3.std():.4f}（预测几乎不动 → 坍缩到均值）")
    L.append(f"- 预测-真值相关系数 r={corr3:+.3f}（≈0 即完全没提取信息）")
    L.append(f"- 模型自报不确定度 σ_rot3 平均 {s3:.3f}（归一化）≈ "
             f"{s3 * phys_half:.3f} rad 物理量 —— σ 大 = 模型自知学不会")
    L.append(f"- 猜均值基线 MAE={np.abs(t3 - t3.mean()).mean()*1000:.1f} mrad，"
             f"模型 MAE={abs_err[:, 5].mean():.1f} mrad → "
             f"{'未跑赢基线（确认躺平）' if abs_err[:, 5].mean() > 0.95*np.abs(t3-t3.mean()).mean()*1000 else '略优于基线'}")
    L.append("")

    # ---------------- 6. 结论区 ----------------
    L.append("## 6. 一句话结论\n")
    ok = [n for j, n in enumerate(PARAM_NAMES)
          if abs_err[:, j].mean() < 0.7 * np.abs(
              R["y_p"][:, j] - R["y_p"][:, j].mean()).mean() * SCALE[j]]
    L.append(f"- 达标参数（{len(ok)}/6）：{', '.join(ok)}")
    L.append(f"- 待修复：{'rot3（雅可比加权/物理加权损失，Week 3）' if 'rot3' not in ok else '无'}")
    L.append(f"- 远距桶 vs 近距桶的 dist 误差对比见 3.1 → 决定是否补远距数据")

    report_fp = os.path.join(outdir, "eval_report.md")
    with open(report_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\nreport → {report_fp}")

    # ---------------- 原始数据落盘 ----------------
    np.savez_compressed(os.path.join(outdir, "eval_raw.npz"),
                        mu_p=R["mu_p"], y_p=R["y_p"], sig_n=R["sig_n"],
                        err_p=err_p, aux=R["aux"], phase_names=R["phase_names"])

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=200)
    cmap = plt.get_cmap("tab10")

    ax = axes[0, 0]   # 标定曲线
    for j, nm in enumerate(PARAM_NAMES):
        s, zz = R["sig_n"][:, j], z[:, j]
        o = np.argsort(s)
        bins = np.array_split(o, 5)
        xs = [s[b].mean() for b in bins]
        ys = [np.sqrt((zz[b] ** 2).mean()) for b in bins]
        ax.plot(xs, ys, "o-", ms=4, lw=1.2, color=cmap(j), label=nm)
    lims = [0.02, 2]
    ax.plot(lims, lims, "k--", lw=0.8, label="ideal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("predicted σ"); ax.set_ylabel("empirical RMS(z)")
    ax.set_title("Uncertainty calibration"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]   # dist 误差 vs 距离桶
    labs, vals, ns = [], [], []
    for k in range(len(edges) - 1):
        m = bi == k
        if m.sum() == 0:
            continue
        labs.append(f"{edges[k]:g}-{edges[k+1]:g}")
        vals.append(abs_err[m, 0].mean())
        ns.append(int(m.sum()))
    bars = ax.bar(labs, vals, color="#4C72B0")
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v, f"n={n}", ha="center",
                va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("detector distance bin (m)")
    ax.set_ylabel("dist MAE (mm)")
    ax.set_title("dist error vs distance")
    ax.grid(alpha=0.3, axis="y", which="both")

    ax = axes[1, 0]   # 各物相 dist 误差
    pm = R["aux"][:, 0].astype(int)
    vals = [abs_err[pm == k, 0].mean() for k in range(len(R["phase_names"]))]
    ax.bar(R["phase_names"], vals, color="#55A868")
    ax.set_yscale("log")
    ax.set_ylabel("dist MAE (mm)")
    ax.set_title("dist error vs phase")
    ax.grid(alpha=0.3, axis="y", which="both")

    ax = axes[1, 1]   # rot3 真值 vs 预测散点
    idx = np.random.default_rng(0).choice(len(t3), 800, replace=False)
    ax.scatter(t3[idx], p3[idx], s=4, alpha=0.4, color="#C44E52")
    ax.plot([-0.2, 0.2], [-0.2, 0.2], "k--", lw=0.8)
    ax.set_xlabel("true rot3 (rad)"); ax.set_ylabel("predicted rot3 (rad)")
    ax.set_title(f"rot3 collapse (r={corr3:+.3f})")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig_fp = os.path.join(outdir, "eval_plots.png")
    fig.savefig(fig_fp, bbox_inches="tight")
    print(f"plots  → {fig_fp}")


if __name__ == "__main__":
    main()
