# -*- coding: utf-8 -*-
"""松绑实验 v2：lr_mult 已否（贴壁率 poni≈0），改测三个新假说。

  H-笼壁(仅rot):  C2 松 poni/rot 笼 trust_k=(0.5,2,2,2,2,2)
  H-未收敛:       C5 iters 400→800
  H-平台期:       C6 链式两段（第一段结果做第二段起点，重暖 nuisance，新 Adam）
  H-信息底:       GT 起点无笼 → poni 能漂多远（似然到底管不管 poni）
"""
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
N, ITERS, NGT = 60, 400, 20

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

rng = np.random.default_rng(7)                    # 与 v1 同样本，可直接对比
ks = sorted(rng.choice(len(ds._sel), N, replace=False).tolist())
data, refs = [], {}
for k in ks:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    true_p = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].cuda()
        mu, lv = model(x)
        pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64),
                             ds.norm_consts)
        sig = np.exp(0.5 * lv[0].cpu().numpy()).astype(np.float64)
    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, os.path.join(ROOT, "data/train_v2"))
    data.append((img, true_p, ph, pred, sig))

PX_UM = refs[data[0][2]].db.pixel * 1e6
CONSTS = refs[data[0][2]].consts
names = ("dist", "poni1", "poni2", "rot1", "rot2")


def run(tag, fn):
    t0 = time.time()
    errs, walls, itrs = [], [], []
    for img, true_p, ph, pred, sig in data:
        r = fn(refs[ph], img, true_p, pred, sig)
        errs.append(np.abs(r["params"] - true_p) * 1000)
        itrs.append(r["iters"])
        pn_fin = DifferentiableRefiner.phys_to_pn(r["params"], CONSTS).numpy()
        pn_ini = DifferentiableRefiner.phys_to_pn(pred, CONSTS).numpy()
        d = np.minimum(pn_fin - (pn_ini - 0.5 * sig - 0.01),
                       (pn_ini + 0.5 * sig + 0.01) - pn_fin)
        walls.append((d < 1e-3).astype(float))
    errs, walls = np.array(errs), np.array(walls)
    wl = "".join(f"{walls[:, i+1].mean()*100:4.0f}" for i in range(4))
    print(f"{tag:24s}" + "".join(f"{errs[:, i].mean():9.3f}" for i in range(5))
          + f"{wl:>8s}  it中位{int(np.median(itrs)):3d}  {time.time()-t0:.0f}s",
          flush=True)
    return errs


print(f"N={N} 像素={PX_UM:.0f}μm  (MAE: mm/mrad; 贴壁%按0.5σ笼)", flush=True)
err0 = np.array([np.abs(p - t) * 1000 for _, t, _, p, _ in data])
print(f"{'NN粗解':24s}" + "".join(f"{err0[:, i].mean():9.3f}" for i in range(5)),
      flush=True)

run("C0 基线 400it",
    lambda r, i, t, p, s: r.refine(i, p, iters=400, sigma_norm=s, trust_k=0.5))
run("C2 松笼(poni/rot x4)",
    lambda r, i, t, p, s: r.refine(i, p, iters=400, sigma_norm=s,
                                   trust_k=(0.5, 2, 2, 2, 2, 2)))
run("C5 长跑 800it",
    lambda r, i, t, p, s: r.refine(i, p, iters=800, sigma_norm=s, trust_k=0.5))


def chained(r, img, true_p, pred, sig):
    r1 = r.refine(img, pred, iters=400, sigma_norm=sig, trust_k=0.5)
    return r.refine(img, r1["params"], iters=400, sigma_norm=sig, trust_k=1.0)


run("C6 链式两段 400+400", chained)

print(f"\n=== GT 起点无笼（N={NGT}）：似然对 poni 的约束力 ===", flush=True)
dr = []
for img, true_p, ph, _, _ in data[:NGT]:
    r = refs[ph].refine(img, true_p, iters=ITERS, sigma_norm=None)
    dr.append(np.abs(r["params"] - true_p) * 1000)
dr = np.array(dr)
print(f"{'参数':6s}{'中位':>8s}{'P90':>8s}   (mm/mrad)")
for i, nm in enumerate(names):
    print(f"{nm:6s}{np.median(dr[:, i]):8.3f}{np.percentile(dr[:, i], 90):8.3f}")
poni_px = (dr[:, 1] + dr[:, 2]) / 2 / PX_UM * 1000
print(f"GT漂移 poni: 中位 {np.median(poni_px):.2f}px  P90 {np.percentile(poni_px, 90):.2f}px"
      f"   [若≪精修误差→优化问题; 若≈→信息/简并底]", flush=True)
