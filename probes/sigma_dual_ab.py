# -*- coding: utf-8 -*-
"""双轨精修 A/B：M1cd（标定σ + refine_dual）vs M1/M1c（sigma_ab 已落盘）。

同 60 张实际工况（train_v3 test, seed=11），同粗解起点 base_v3/last.pt。
M1cd = refine_dual(sigma_norm=σ·T, trust_k=1.0, soft_prior=0.1, adopt_nats=1.0)
     = 笼内轨A（标定σ软先验）+ 同起点拆笼轨B，纯似然好 >1 nat 才采纳 B。

输出：三配方对比表 + 采纳率 + 误伤清单（采纳 B 但 dist 误差比 M1c 差 >1mm）。
"""
import glob
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
N = 60
CKPT = os.path.join(ROOT, "runs/base_v3/last.pt")

# ---- 与 sigma_ab 完全相同的抽样 ----
ds = TorchFAIDataset(os.path.join(ROOT, "data/train_v3"), split="test")
aux = ds._maux[ds._sel]
mask = (aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
idx = np.where(mask)[0]
sel = sorted(np.random.default_rng(11).choice(idx, min(N, len(idx)), replace=False).tolist())

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"])
model.eval().cuda()

T = np.array([json.load(open(os.path.join(ROOT, "runs/base_v3/sigma_T.json"),
                            encoding="utf-8"))["T"][n] for n in
              ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")])

refs, samples = {}, []
for k in sel:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    tp = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
        mu, lv = model(x)
        pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
        sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v3"))
    samples.append((img, tp, ph, pred, sig))

px_mm = refs[samples[0][2]].db.pixel * 1000.0
res, adopts, gaps = [], [], []
t0 = time.time()
for i, (img, tp, ph, pred, sig) in enumerate(samples):
    r = refs[ph].refine_dual(img, pred, sigma_norm=sig * T, trust_k=1.0,
                             soft_prior=0.1, iters=400, adopt_nats=1.0)
    res.append(np.abs(r["params"] - tp) * 1000)
    adopts.append(r["dual_adopt"])
    gaps.append(r["pure_A"] - r["pure_B"])          # >0 = B 更好
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(samples)}  采纳 {sum(adopts)}  {time.time()-t0:.0f}s",
              flush=True)

res = np.array(res)
adopts = np.array(adopts)
gaps = np.array(gaps)

# ---- 读 sigma_ab 的 M1/M1c 结果做配对 ----
prev_fp = sorted(glob.glob(os.path.join(ROOT, "probes", "results", "sigma_ab_*.npz")))[-1]
prev = np.load(prev_fp)
e_m1, e_m1c = prev["M1"], prev["M1c"]
print(f"\n配对基准：{os.path.basename(prev_fp)}（同 60 张）")
print(f"双轨采纳率：{adopts.sum()}/{len(adopts)}（纯似然差 中位 "
      f"{np.median(gaps[adopts]) if adopts.any() else 0:.2f} nats）\n")

hdr = (f"{'配方':6s}{'distMAE':>9s}{'dist中位':>9s}{'<0.5mm':>8s}"
       f"{'poni中位px':>11s}{'poni_p75':>9s}{'rot2中位':>9s}{'>5mm尾部':>9s}")
print(hdr)


def show(name, e):
    poni_px = (e[:, 1] + e[:, 2]) / 2 / px_mm
    print(f"{name:6s}{e[:,0].mean():9.3f}{np.median(e[:,0]):9.3f}"
          f"{(e[:,0]<0.5).mean()*100:7.1f}%{np.median(poni_px):11.2f}"
          f"{np.percentile(poni_px,75):9.2f}{np.median(e[:,4]):9.3f}"
          f"{(e[:,0]>5).sum():9d}")


show("M1", e_m1)
show("M1c", e_m1c)
show("M1cd", res)

# ---- 误伤检查：采纳 B 的样本里，dist 误差比 M1c 差多少 ----
ai = np.where(adopts)[0]
if len(ai):
    d = res[ai, 0] - e_m1c[ai, 0]                  # >0 = 双轨比纯标定差
    print(f"\n采纳双轨的 {len(ai)} 张：dist 误差 vs M1c  "
          f"中位 {np.median(d):+.3f}mm  最差 {d.max():+.3f}mm  变差>1mm: {(d>1).sum()} 张")
    for j in np.where(d > 1)[0]:
        k = ai[j]
        print(f"  误伤样本 #{k}: M1c {e_m1c[k,0]:.2f} → M1cd {res[k,0]:.2f}mm"
              f"（B-A nats {gaps[k]:.2f}）")
# 未采纳但 M1c 已差的样本：B 轨没救回（信息量天花板/假谷）
ni = np.where(~adopts)[0]
bad_m1c = ni[e_m1c[ni, 0] > 5]
print(f"M1c 中 >5mm 的 {len(bad_m1c)} 张里，双轨救回 "
      f"{(res[bad_m1c,0]<=5).sum() if len(bad_m1c) else 0} 张")

np.savez_compressed(os.path.join(ROOT, "probes", "results",
                                 f"sigma_dual_{time.strftime('%Y%m%d_%H%M%S')}.npz"),
                    M1cd=res, adopts=adopts, gaps=gaps)
print(f"\n落盘 probes/results/sigma_dual_*.npz  总耗时 {time.time()-t0:.0f}s")
