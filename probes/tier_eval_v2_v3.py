# -*- coding: utf-8 -*-
"""base_v2 vs base_v3 分档评测（论文口径版）。

档位定义（与 probes/_practice_subset.py / _map_prior_ab.py 一致）：
  实际工况档 practice : n_vis>=6 且 peak_scale>=10000 且 bs_r<0（无挡光盘）
  少环档   few       : n_vis<=3
  全体档   all

两个数据集口径：
  v3test  = data/train_v3 split=test（4800）→ **同卷配对对比**，v2/v3 用同一批图
  v2test  = data/train_v2 split=test（3900）→ 与历史数字（poni 3.34px 等）对齐

阶段：
  --stage nn      只跑 NN 粗解层（全量、快）：六参数物理 MAE + σ 标定
  --stage refine  跑精修层（抽样）：现行 C0 配方（硬笼 0.5σ, iters=400）

输出：probes/results/tier_eval_<stage>_<ts>.csv / .md
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "probes", "results")
os.makedirs(OUTDIR, exist_ok=True)

UNITS = ("mm", "mm", "mm", "mrad", "mrad", "mrad")
SCALE = (1000.0,) * 6
PNAMES = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")


def tiers_of(aux):
    n_vis = aux[:, 1].astype(int)
    scale = aux[:, 3]
    bs = aux[:, 4]
    return {
        "all": np.ones(len(aux), bool),
        "practice": (n_vis >= 6) & (scale >= 10000) & (bs < 0),
        "few": n_vis <= 3,
    }


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ResNet18Geo()
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    m.eval().cuda()
    return m, ck


@torch.no_grad()
def nn_predict(model, ds, batch=64):
    """返回 (mu_p, sig_n, y_p, aux)。逐样本读 mmap，避免 DataLoader 多进程。"""
    idx = ds._sel
    mu_l, sig_l, y_l = [], [], []
    for s in range(0, len(idx), batch):
        sl = idx[s:s + batch]
        imgs = np.stack([np.asarray(ds._mimgs[int(g)]) for g in sl]).astype(np.float32)
        x = torch.from_numpy(np.log1p(imgs)).unsqueeze(1).cuda()
        with torch.autocast("cuda", enabled=True):
            mu, lv = model(x)
        mu_l.append(mu.float().cpu().numpy())
        sig_l.append(np.exp(0.5 * lv.float().cpu().numpy()))
        y_l.append(ds._mparams[sl].astype(np.float64))
    mu_n = np.concatenate(mu_l)
    sig_n = np.concatenate(sig_l)
    # 注意：ds._mparams 已是物理量，不能再过 norm_to_param（那是逆变换）
    y_p = np.concatenate(y_l)
    mu_p = norm_to_param(mu_n, ds.norm_consts)
    return mu_p, sig_n, y_p, ds._maux[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="nn", choices=["nn", "refine"])
    ap.add_argument("--n-refine", type=int, default=60)
    ap.add_argument("--ckpts", default="runs/base_v2/best.pt,runs/base_v3/best.pt,runs/base_v3/last.pt")
    ap.add_argument("--datasets", default="train_v3,train_v2")
    a = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_fp = os.path.join(OUTDIR, f"tier_eval_{a.stage}_{ts}.csv")
    md_fp = os.path.join(OUTDIR, f"tier_eval_{a.stage}_{ts}.md")
    rows = []
    md = [f"# 分档评测 {a.stage} — {time.strftime('%Y-%m-%d %H:%M')}\n"]

    ckpts = [c.strip() for c in a.ckpts.split(",") if c.strip()]
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]

    for dname in datasets:
        droot = os.path.join(ROOT, "data", dname)
        ds = TorchFAIDataset(droot, split="test")
        aux = ds._maux[ds._sel]
        tm = tiers_of(aux)
        print(f"\n########## 数据集 {dname} test N={len(ds._sel)} ##########", flush=True)
        for k, m in tm.items():
            print(f"  档 {k:9s} N={int(m.sum()):5d} ({m.sum()/len(m)*100:5.1f}%)", flush=True)
        md.append(f"\n## 数据集 {dname}（test, N={len(ds._sel)}）\n")
        md.append("| 档 | N | 占比 |")
        md.append("|---|---|---|")
        for k, m in tm.items():
            md.append(f"| {k} | {int(m.sum())} | {m.sum()/len(m)*100:.1f}% |")
        md.append("")

        for ck_rel in ckpts:
            ck_fp = os.path.join(ROOT, ck_rel)
            if not os.path.isfile(ck_fp):
                print(f"  [跳过] 缺 {ck_rel}")
                continue
            model, ck = load_model(ck_fp)
            name = ck_rel.replace("runs/", "").replace("/", "_").replace(".pt", "")
            t0 = time.time()
            mu_p, sig_n, y_p, aux_ds = nn_predict(model, ds)
            err = (mu_p - y_p) * SCALE
            # 用归一化真值/预测算 z（与 eval.py 口径一致：z=(y_n-mu_n)/sigma_n）
            from torchfai.dataset import param_to_norm
            y_n = param_to_norm(y_p, ds.norm_consts)
            mu_n = param_to_norm(mu_p, ds.norm_consts)
            z = (y_n - mu_n) / sig_n
            print(f"\n  == 模型 {name} (epoch={ck.get('epoch')}, "
                  f"best_val={ck.get('best_val')}) 推理 {time.time()-t0:.0f}s ==", flush=True)

            for tk, m in tm.items():
                if m.sum() == 0:
                    continue
                ae = np.abs(err[m])
                row = dict(dataset=dname, model=name, tier=tk, N=int(m.sum()))
                for j, pn in enumerate(PNAMES):
                    row[f"{pn}_mae"] = round(float(ae[:, j].mean()), 4)
                    row[f"{pn}_med"] = round(float(np.median(ae[:, j])), 4)
                row["dist_lt05_pct"] = round(float((ae[:, 0] < 0.5).mean() * 100), 2)
                row["dist_lt1_pct"] = round(float((ae[:, 0] < 1.0).mean() * 100), 2)
                for j, pn in enumerate(PNAMES):
                    row[f"z_rms_{pn}"] = round(float(np.sqrt((z[m, j] ** 2).mean())), 4)
                    row[f"cov68_{pn}"] = round(float((np.abs(z[m, j]) < 1).mean() * 100), 2)
                rows.append(row)
                print(f"    档 {tk:9s} N={int(m.sum()):5d}  dist MAE={row['dist_mae']:8.3f}mm "
                      f"med={row['dist_med']:7.3f}  <0.5mm={row['dist_lt05_pct']:5.1f}%  "
                      f"poni1 MAE={row['poni1_mae']:7.3f}mm  poni2 MAE={row['poni2_mae']:7.3f}mm  "
                      f"rot2 MAE={row['rot2_mae']:7.3f}mrad  zRMS(dist)={row['z_rms_dist']:5.2f}",
                      flush=True)
            if a.stage == "refine":
                refs = {}
                px_um = None
                for tk, m in tm.items():
                    if tk == "few":
                        continue
                    sel = np.where(m)[0]
                    rng = np.random.default_rng(11 if tk == "practice" else 7)
                    take = rng.choice(sel, min(a.n_refine, len(sel)), replace=False)
                    for j in take:
                        g = int(ds._sel[j])
                        img = np.asarray(ds._mimgs[g])
                        tp = ds._mparams[g].astype(np.float64)
                        ph = PHASE_NAMES[int(ds._maux[g][0])]
                        if ph not in refs:
                            refs[ph] = DifferentiableRefiner(ph, droot)
                        if px_um is None:
                            px_um = refs[ph].db.pixel * 1e6
                        with torch.no_grad():
                            x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
                            mu, lv = model(x)
                            pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
                            sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
                        r = refs[ph].refine(img, pred, iters=400, sigma_norm=sig, trust_k=0.5)
                        e = np.abs(r["params"] - tp) * 1000
                        # dist/poni: m→mm；rot: rad→mrad（同一 1000 因子）
                        row = dict(dataset=dname, model=name, tier=tk, N=1)
                        for jj, pn in enumerate(PNAMES):
                            row[f"{pn}_err"] = float(e[jj])
                        rows.append(row)
                        print(f"      refine {tk} {name} sample err dist={e[0]:.3f}mm "
                              f"poni={(e[1]+e[2])/2:.3f}mm rot2={e[4]:.3f}mrad", flush=True)
                    print(f"    refine 档 {tk} 完成", flush=True)

    if rows:
        with open(csv_fp, "w", newline="", encoding="utf-8") as f:
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        md.append(f"\n原始数据 → {csv_fp}\n")
        with open(md_fp, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
    print(f"\nCSV → {csv_fp}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
