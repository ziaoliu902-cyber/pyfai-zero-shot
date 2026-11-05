# -*- coding: utf-8 -*-
"""松绑实验：poni/rot 精修受限的根因定位 + 磨刀包候选配方验证。

四问：
  1) 精修终点是否撞笼壁（trust_k·σ+0.01）？撞 = 笼子锁死
  2) lr_mult 放大 poni/rot 学习率能否降误差？
  3) 只松 poni/rot 的笼（trust_k 向量）能否降误差？
  4) GT 起点无笼精修，poni 能漂多远？（图像信息底）
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
N = 60
ITERS = 400                      # 与 refine_v2 快速档一致
NGT = 20                         # GT 起点对照样本数

CONFIGS = {
    "C0 基线(=refine_v2)": dict(),
    "C1 lr_mult poni/rot x3": dict(lr_mult=(1, 3, 3, 3, 3, 1)),
    "C2 松笼 poni/rot x4": dict(trust_k=(0.5, 2.0, 2.0, 2.0, 2.0, 2.0)),
    "C3 lr+松笼": dict(lr_mult=(1, 3, 3, 3, 3, 1),
                       trust_k=(0.5, 2.0, 2.0, 2.0, 2.0, 2.0)),
}

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

rng = np.random.default_rng(7)
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

PX_UM = refs[data[0][2]].db.pixel * 1e6          # 像素尺寸(μm)
CONSTS = refs[data[0][2]].consts
print(f"N={N} iters={ITERS}  像素={PX_UM:.1f}μm  "
      f"相机全高 Hm={CONSTS['Hm']*1000:.0f}mm")

names = ("dist", "poni1", "poni2", "rot1", "rot2")


def cage_stats(r, pred, sig, tk=0.5):
    """精修终点离笼壁多远：返回各参数 [0,1]（1=贴壁）。"""
    pn_fin = DifferentiableRefiner.phys_to_pn(r["params"], CONSTS).numpy()
    pn_ini = DifferentiableRefiner.phys_to_pn(pred, CONSTS).numpy()
    tkv = (np.full(6, tk) if np.isscalar(tk)
           else np.asarray(tk, float).ravel())
    margin = tkv * sig + 0.01
    lo, hi = pn_ini - margin, pn_ini + margin
    d = np.minimum(pn_fin - lo, hi - pn_fin)     # 距两端的最小距离(norm)
    return (d < 1e-3).astype(float)              # 贴壁=1


err0 = np.array([np.abs(p - t) * 1000 for _, t, _, p, _ in data])
print(f"\n{'NN粗解':22s}" + "".join(f"{nm:>9s}" for nm in names)
      + f"{'贴壁%':>7s}" + "   (MAE: mm / mrad)")
print(f"{'':22s}" + "".join(f"{err0[:, i].mean():9.3f}" for i in range(5)))

t_all = time.time()
for tag, kw in CONFIGS.items():
    errs, walls = [], []
    for img, true_p, ph, pred, sig in data:
        r = refs[ph].refine(img, pred, iters=ITERS, sigma_norm=sig,
                            trust_k=0.5, **kw)
        errs.append(np.abs(r["params"] - true_p) * 1000)
        walls.append(cage_stats(r, pred, sig,
                                kw.get("trust_k", 0.5)))
    errs, walls = np.array(errs), np.array(walls)
    wl = "".join(f"{walls[:, i+1].mean()*100:4.0f}" for i in range(4))
    st = (errs[:, 0] > 5).mean() * 100
    print(f"{tag:22s}" + "".join(f"{errs[:, i].mean():9.3f}" for i in range(5))
          + f"{wl:>8s}  stuck={st:.0f}%  {time.time()-t_all:.0f}s")

# ---- GT 起点对照：无笼，图像信息底 ----
print(f"\n=== GT 起点无笼精修（N={NGT}，信息底）===")
dr = []
for img, true_p, ph, _, _ in data[:NGT]:
    r = refs[ph].refine(img, true_p, iters=ITERS, sigma_norm=None)
    dr.append(np.abs(r["params"] - true_p) * 1000)
dr = np.array(dr)
print(f"{'参数':6s}{'漂移中位':>9s}{'漂移P90':>9s}   (mm / mrad)")
for i, nm in enumerate(names):
    print(f"{nm:6s}{np.median(dr[:, i]):9.3f}{np.percentile(dr[:, i], 90):9.3f}")
print(f"poni 换算像素: 中位 {np.median((dr[:,1]+dr[:,2])/2)/PX_UM*1000:.2f}px"
      f"  P90 {np.percentile((dr[:,1]+dr[:,2])/2,90)/PX_UM*1000:.2f}px")
