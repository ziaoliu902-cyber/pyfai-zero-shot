# -*- coding: utf-8 -*-
"""_causal_test.py：孪生-移植因果定位（v4w3 重跑）。
真实帧 Δ≠0 而孪生 Δ≈0 → 失配在图外观而非几何；
再移植真实环强度剖面到孪生：若移植后 Δ≈真实 → 强度剖面为病根。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI

from real_refine_v6 import load_px_nn, to_nn_input_wide, match_gain
from torchfai.config import SimConfig
from torchfai.phases import PHASE_NAMES
from torchfai.dataset import TorchFAIDataset as _T
from torchfai.dataset import norm_to_param
from torchfai.simulator import get_scaled_detector, synth_sample

m, dev, consts = load_px_nn(r"runs\v4w3\best.pt")
consts = _T(os.path.join(ROOT, "data", "train_v4w3"), "val").norm_consts
m.eval()
LAM = 0.972386
K = 2.425
PXEFF = 172e-6 / (521.0 / 2527)


def coarse(canvas):
    x = torch.from_numpy(np.log1p(canvas[None, None].astype(np.float32))).to(dev)
    with torch.no_grad():
        mu, _ = m(x, torch.tensor([LAM], device=dev),
                  torch.tensor([K], device=dev))
    return float(norm_to_param(mu.double().cpu().numpy(), consts, K)[0, 0])


def radial(canvas, pr, pc, nb=300):
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - pr, xx - pc)
    idx = np.clip((r / r.max() / nb).astype(int), 0, nb - 1)
    prof = np.zeros(nb)
    cnt = np.zeros(nb)
    np.add.at(prof, idx.ravel(), canvas.ravel())
    np.add.at(cnt, idx.ravel(), 1)
    prof /= np.maximum(cnt, 1)
    return prof, idx


for d in (250, 400):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    D = p.dist
    wide, k, extra = to_nn_input_wide(img)
    wide = match_gain(wide)
    pr = p.poni1 / PXEFF
    pc = (p.poni2 - extra) / PXEFF

    db = get_scaled_detector("Pilatus_1M", 2, K)
    Hm, Wm = db.extent
    rng = np.random.default_rng(7)
    g = dict(dist=D, poni1=Hm * (pr / 521), poni2=Wm * (pc / 490),
             rot1=0.0, rot2=0.0, rot3=0.0)
    cfg = SimConfig()
    cfg.wavelength = LAM * 1e-10
    s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                     pixel_scale=K, geom=g)
    twin = s["img"].astype(np.float64)

    prof_r, _ = radial(wide.astype(np.float64), pr, pc)
    prof_t, idx = radial(twin, Hm * (pr / 521), Wm * (pc / 490))
    # 增益图 = 真实剖面/孪生剖面（防0除，平滑clip）
    ratio = prof_r / np.maximum(prof_t, 0.05 * prof_t.max())
    ratio = np.clip(ratio, 0.1, 10.0)
    trans = twin * ratio[idx]
    trans = np.clip(trans / max(trans.max(), 1) * 65535, 0, 65535)

    b_twin = (coarse(twin) - D) * 1000
    b_trans = (coarse(trans) - D) * 1000
    b_real = (coarse(wide) - D) * 1000
    print(f"ceria{d}: 孪生 {b_twin:+.1f} → 移植后 {b_trans:+.1f}mm"
          f"   （真实帧 {b_real:+.1f}）")
print("\n移植后≈真实 → 环强度/背景剖面 = 病根（训练侧放宽幅度随机化即治）")
