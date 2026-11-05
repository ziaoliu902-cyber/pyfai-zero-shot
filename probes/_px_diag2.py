# -*- coding: utf-8 -*-
"""px_v2 二次诊断：sim 域 vs 真实域分裂定位 + alumina 输入管线体检。"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio

from real_refine_v6 import to_nn_input_wide
from torchfai.config import SimConfig
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.phases import PHASE_NAMES
from torchfai.simulator import get_scaled_detector, synth_sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(os.path.join(ROOT, "runs/px_v2/best.pt"),
                map_location="cpu", weights_only=False)
m = ResNet18Geo(lam_cond=True, px_cond=True).eval()
m.load_state_dict(ck["model"])
m.to(dev)
consts = TorchFAIDataset(os.path.join(ROOT, "data/train_px1"), "val").norm_consts


def pred(img, lam, k):
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].to(dev)
        mu, _ = m(x, torch.tensor([lam], device=dev),
                  torch.tensor([k], device=dev))
    return norm_to_param(mu.float().cpu().numpy(), consts, k)[0]


print("== 1. sim wide direct (k=2.425, ceria+Al2O3, two dists) ==")
cfg = SimConfig()
cfg.wavelength = 9.72386e-11
db = get_scaled_detector("Pilatus_1M", 2, 2.425)
Hm, Wm = db.extent
rng = np.random.default_rng(7)
for phase in ("CeO2", "Al2O3"):
    for dist in (0.15, 0.45):
        pi = PHASE_NAMES.index(phase)
        g = dict(dist=dist, poni1=Hm * 0.5, poni2=Wm * 0.5,
                 rot1=0.0, rot2=0.0, rot3=0.0)
        s = synth_sample(rng, cfg, db, phase_idx=pi, pixel_scale=2.425, geom=g)
        p = pred(s["img"], 0.972386, 2.425)
        print(f"   {phase} @{int(dist*1000)}mm: pred={p[0]*1000:7.1f}mm "
              f"d={(p[0]-dist)*1000:+7.1f}mm")

print("== 2. sim broad Al2O3 (broad regime, k=1, alumina-like GT) ==")
db1 = get_scaled_detector("Pilatus_1M", 2, 1.0)
Hm1, Wm1 = db1.extent
pi = PHASE_NAMES.index("Al2O3")
cfg_b = SimConfig()
cfg_b.wavelength = 9.72386e-11
cfg_b.peak_sigma_broad_frac = 1.0
g = dict(dist=0.117, poni1=Hm1 * 0.5, poni2=Wm1 * 0.5,
         rot1=0.0, rot2=0.0, rot3=0.0)
rng_b = np.random.default_rng(21)
for trial in range(3):
    s = synth_sample(rng_b, cfg_b, db1, phase_idx=pi, pixel_scale=1.0, geom=g)
    p = pred(s["img"], 0.972386, 1.0)
    print(f"   trial{trial}: GT=117mm pred={p[0]*1000:7.1f}mm "
          f"d={(p[0]-0.117)*1000:+7.1f}mm")

print("== 3. real 6M wide 150mm retest ==")
img = fabio.open(os.path.join(ROOT, "data/real/MX_ceria/ceria_150_1_0001.cbf")).data
wide, k, extra = to_nn_input_wide(img)
p = pred(wide, 0.972386, k)
print(f"   pred={p[0]*1000:.1f}mm (GT=150)")

print("== 4. canvas stats sim vs real (gap cadence autoref) ==")
rng2 = np.random.default_rng(9)
g = dict(dist=0.15, poni1=Hm * 0.5, poni2=Wm * 0.5,
         rot1=0.0, rot2=0.0, rot3=0.0)
pi = PHASE_NAMES.index("CeO2")
s = synth_sample(rng2, cfg, db, phase_idx=pi, pixel_scale=2.425, geom=g)
sim = s["img"]
for nm, a in (("sim", sim), ("real", wide)):
    dead = (a == 0)
    prof = dead.mean(axis=0)
    z = prof - prof.mean()
    ac = np.correlate(z, z, "full")
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] if ac[0] > 0 else 1)
    peaks = np.where((ac[1:-1] > 0.3) & (ac[1:-1] > ac[:-2]) &
                     (ac[1:-1] >= ac[2:]))[0][:3] + 1
    print(f"   {nm}: shape={a.shape} max={a.max()} mean={a.astype(np.float64).mean():.2f} "
          f"zerofrac={dead.mean():.3f} cadence_peaks={peaks}")

print("== 5. real alumina frame stats ==")
al = fabio.open(os.path.join(ROOT, "data/real/Al2O3/Al2O3_cair_0001.cbf")).data
print(f"   raw: shape={al.shape} dtype={al.dtype} max={al.max()} "
      f"zerofrac={(al == 0).mean():.3f} mean={al.astype(np.float64).mean():.1f}")
