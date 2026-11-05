# -*- coding: utf-8 -*-
"""实际工况档【全量】精修评测 —— 消除 n=30 抽样噪声的唯一办法。

背景（2026-10-26 实测）：该档 poni/rot 误差是重尾分布。n=30 抽样得到的中位数
在 3.3px（运气好）与 15.4px（同一配方）之间跳，重复抽样不稳定；同抽样复现与
落盘 NPZ 逐位相同（max diff 0.0），证明差异纯粹来自抽样。
→ 该档所有结论必须用全量池（train_v2: 315 张 / train_v3: 394 张）。

组合：模型 {base_v2, base_v3} × 配方 {C0 硬笼0.5σ, M1 软先验 w=0.1}
抽样顺序：样本在外、组合在内 → 中途中断也是配对完整、可增量使用。

输出：probes/results/practice_fullpool_<dataset>_<tier>_<ts>.csv（逐样本，逐行 flush）
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

RECIPES = {
    "C0": dict(trust_k=0.5),
    "M1": dict(trust_k=1.0, soft_prior=0.1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="train_v3")
    ap.add_argument("--ckpts", default="runs/base_v2/best.pt,runs/base_v3/last.pt")
    ap.add_argument("--recipes", default="C0,M1")
    ap.add_argument("--limit", type=int, default=0, help="0=全池")
    ap.add_argument("--tier", default="practice", choices=["practice", "all"])
    a = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_fp = os.path.join(OUTDIR, f"practice_fullpool_{a.dataset}_{a.tier}_{ts}.csv")

    droot = os.path.join(ROOT, "data", a.dataset)
    ds = TorchFAIDataset(droot, split="test")
    aux = ds._maux[ds._sel]
    n_vis = aux[:, 1].astype(int)
    if a.tier == "practice":
        mask = (n_vis >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
    else:
        mask = np.ones(len(aux), bool)
    sel = np.where(mask)[0]
    if a.limit:
        sel = sel[:a.limit]
    print(f"dataset={a.dataset} tier={a.tier} pool={len(sel)} -> {csv_fp}", flush=True)

    ck_paths = [c.strip() for c in a.ckpts.split(",") if c.strip()]
    recipe_names = [r.strip() for r in a.recipes.split(",") if r.strip()]

    models = {}
    for c in ck_paths:
        fp = os.path.join(ROOT, c)
        if not os.path.isfile(fp):
            print(f"[skip] missing {c}")
            continue
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        m = ResNet18Geo()
        m.load_state_dict(ck["model"] if "model" in ck else ck)
        m.eval().cuda()
        models[c] = (m, ck)
    print("models:", list(models), " recipes:", recipe_names, flush=True)

    refs = {}
    fh = open(csv_fp, "w", newline="", encoding="utf-8")
    fw = csv.writer(fh)
    fw.writerow(["idx", "model", "recipe", "dist_mm", "poni1_mm", "poni2_mm",
                 "rot1_mrad", "rot2_mrad", "rot3_mrad", "n_vis", "scale", "phase"])
    t0 = time.time()
    for i, j in enumerate(sel):
        g = int(ds._sel[j])
        img = np.asarray(ds._mimgs[g])
        tp = ds._mparams[g].astype(np.float64)
        ph = PHASE_NAMES[int(ds._maux[g][0])]
        if ph not in refs:
            refs[ph] = DifferentiableRefiner(ph, droot)
        for cname, (m, ck) in models.items():
            with torch.no_grad():
                x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
                mu, lv = m(x)
                pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
                sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
            for rname in recipe_names:
                r = refs[ph].refine(img, pred, iters=400, sigma_norm=sig, **RECIPES[rname])
                e = np.abs(r["params"] - tp) * 1000.0   # m->mm, rad->mrad
                fw.writerow([int(g), cname, rname] + [round(float(v), 5) for v in e]
                            + [int(n_vis[j]), float(aux[j, 3]), ph])
        if (i + 1) % 10 == 0:
            fh.flush()
            el = time.time() - t0
            rate = el / (i + 1)
            print(f"  {i+1}/{len(sel)}  {el/60:.1f}min  {rate:.1f}s/img  "
                  f"ETA {(len(sel)-i-1)*rate/60:.0f}min", flush=True)
    fh.flush()
    fh.close()
    print(f"\nCSV -> {csv_fp}\nALL DONE")


if __name__ == "__main__":
    main()
