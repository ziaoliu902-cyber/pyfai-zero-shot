# -*- coding: utf-8 -*-
"""拆解 v3 vs v4b：按 dist<=1.0 / >1.0 分段，看"实际工况"档的胜败在哪。

问题：v4b 用 finetune 常数解码后，实际工况档仍略输 v3。为什么？
假设：微调集 60% practice 段 dist 全在 [0.1,1.0]（中位 0.136m），
      而 finetune_v1/meta.json 把 config.dist_range 写死 [0.1,1.0]，
      regular 段 8000 张里有 399 张 dist>1.0m 的标签越界 (y>1)。
      → 微调后 dist 头学的是另一套编码，且被越界标签污染。
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ft = TorchFAIDataset(os.path.join(ROOT, "data", "finetune_v1"), split="train")
v2 = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")

aux = v2._maux[v2._sel]
print("=== aux 列语义自检（train_v2 test）===")
for c, nm in enumerate(["phase_idx", "n_vis", "qmax", "scale/imax", "bs_r/hump"]):
    col = aux[:, c]
    print(f"  col{c} {nm:12s} min={col.min():.4f} max={col.max():.4f} "
          f"median={np.median(col):.4f}")
print("  -> col1=n_vis? col3=imax? col4=hump? 由下方分布间接验证\n")

nvis, imax, hump = aux[:, 1], aux[:, 3], aux[:, 4]
prac = (nvis >= 6) & (imax >= 10000) & (hump < 0)
truth = v2._mparams[v2._sel]
d = truth[:, 0]
print(f"=== 实际工况档(315)的 dist 分布 ===")
print(f"  min={d[prac].min():.4f} max={d[prac].max():.4f} median={np.median(d[prac]):.4f}")
print(f"  >1.0m 的占比: {(d[prac]>1.0).mean()*100:.1f}%   "
      f">0.77m: {(d[prac]>0.77).mean()*100:.1f}%")
print(f"  全体 test 的 dist 中位 {np.median(d):.4f}  >1.0m: {(d>1.0).mean()*100:.1f}%\n")

# 训练集侧：finetune 各段的 dist 分布
pf = np.load(os.path.join(ROOT, "data", "finetune_v1", "params.npy"), mmap_mode="r")
af = np.load(os.path.join(ROOT, "data", "finetune_v1", "aux.npy"), mmap_mode="r")
prac_rows = np.arange(0, 12000)
reg_rows = np.arange(12000, 20000)
print("=== finetune_v1 各段 ===")
for nm, rows in (("practice", prac_rows), ("regular", reg_rows)):
    dd = np.asarray(pf[rows])[:, 0]
    aa = np.asarray(af[rows])
    mp = (aa[:, 1] >= 6) & (aa[:, 3] >= 10000) & (aa[:, 4] < 0)
    print(f"  {nm:9s} N={len(dd):6d} dist中位={np.median(dd):.4f} max={dd.max():.3f} "
          f">1.0m:{(dd>1.0).mean()*100:4.1f}%  实际工况画像:{mp.mean()*100:5.1f}%")
# 越界标签
y = np.asarray(pf)[:, 0]
yn = (np.log10(y) + 1.0) / 1.0 * 2.0 - 1.0
print(f"  标签越界(y_dist>1) 行数: {(yn>1).sum()}  (全部来自 regular 段)")

MODELS = [("base_v3", "runs/base_v3/best.pt", v2.norm_consts),
          ("base_v4b", "runs/base_v4b/best.pt", ft.norm_consts)]
PX_UM = 344.0
TIERS = [("全体", np.ones(len(v2._sel), bool)),
         ("实际工况", prac),
         ("实际工况&dist<=1m", prac & (d <= 1.0)),
         ("实际工况&dist>1m", prac & (d > 1.0)),
         ("远距>0.77m", d > 0.77)]
res = {}
for tag, rel, nc in MODELS:
    fp = os.path.join(ROOT, rel)
    if not os.path.exists(fp):
        continue
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval().cuda()
    E = np.zeros((len(v2._sel), 6))
    t0 = time.time()
    idx_all = np.arange(len(v2._sel))
    for s in range(0, len(idx_all), 64):
        ks = idx_all[s:s + 64]
        xb = np.stack([np.log1p(np.asarray(v2._mimgs[int(v2._sel[k])],
                                           dtype=np.float32)) for k in ks])
        with torch.no_grad():
            x = torch.from_numpy(xb)[:, None].cuda()
            mu, lv = model(x)
        for j, k in enumerate(ks):
            E[k] = np.abs(norm_to_param(mu[j].cpu().numpy().astype(np.float64), nc) - truth[k])
    res[tag] = E
    print(f"\n===== {tag}  ({time.time()-t0:.0f}s) =====")
    print(f"{'档':20s}{'N':>5s}{'distMAE':>9s}{'dist中位':>9s}{'尾>10mm':>9s}{'poni中位px':>11s}")
    for tname, m in TIERS:
        if m.sum() == 0:
            continue
        e = E[m] * 1000
        poni_px = np.median((e[:, 1] + e[:, 2]) / 2 / PX_UM * 1000)
        print(f"{tname:20s}{m.sum():5d}{np.mean(e[:,0]):9.2f}{np.median(e[:,0]):9.3f}"
              f"{(e[:,0]>10).mean()*100:8.1f}%{poni_px:11.2f}")

if len(res) == 2:
    a, b = res["base_v3"] * 1000, res["base_v4b"] * 1000
    print("\n=== v4b 相对 v3 的 dist 中位变化 ===")
    for tname, m in TIERS:
        if m.sum() == 0:
            continue
        print(f"  {tname:20s} v3={np.median(a[m,0]):8.3f}  v4b={np.median(b[m,0]):8.3f}  "
              f"Δ={np.median(b[m,0])-np.median(a[m,0]):+8.3f} mm")
