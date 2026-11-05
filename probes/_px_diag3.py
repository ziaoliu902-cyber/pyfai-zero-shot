# -*- coding: utf-8 -*-
"""px_v2 三诊：真实宽画布输入变换消融——强度归一 vs 缝隙填补，谁救得回。"""
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


def pred(img, lam=0.972386, k=2.425):
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None].to(dev)
        mu, _ = m(x, torch.tensor([lam], device=dev),
                  torch.tensor([k], device=dev))
    return norm_to_param(mu.float().cpu().numpy(), consts, k)[0]


def inpaint_gaps(a, dead):
    """沿行方向线性内插填补 dead 列区段（含端部延伸）。"""
    out = a.astype(np.float64).copy()
    for r in range(out.shape[0]):
        row, d = out[r], dead[r]
        if not d.any():
            continue
        good = np.where(~d)[0]
        if len(good) == 0:
            continue
        row[d] = np.interp(np.where(d)[0], good, row[good])
    return out


# 参考亮度：sim 宽视域的典型 p99（用一张固定合成图标定）
cfg = SimConfig()
cfg.wavelength = 9.72386e-11
db = get_scaled_detector("Pilatus_1M", 2, 2.425)
Hm, Wm = db.extent
rng = np.random.default_rng(3)
s = synth_sample(rng, cfg, db, phase_idx=PHASE_NAMES.index("CeO2"),
                 pixel_scale=2.425,
                 geom=dict(dist=0.15, poni1=Hm * 0.5, poni2=Wm * 0.5,
                           rot1=0, rot2=0, rot3=0))
ref_p99 = np.percentile(s["img"], 99.9)
print(f"sim 参考 p99.9 = {ref_p99:.0f}")

for gt_mm, fn in ((150, "ceria_150_1_0001.cbf"),
                  (450, "ceria_450_1_0001.cbf")):
    img = fabio.open(os.path.join(ROOT, "data/real/MX_ceria", fn)).data
    wide, k, extra = to_nn_input_wide(img)
    dead = wide == 0
    p99 = np.percentile(wide, 99.9)
    gain = ref_p99 / max(p99, 1e-6)
    variants = {
        "a_raw": wide,
        "b_xgain": wide * gain,
        "c_inpaint": inpaint_gaps(wide, dead),
        "d_both": inpaint_gaps(wide, dead) * gain,
        "e_logp1_clip": wide,          # 占位，同 a
    }
    print(f"-- {fn} GT={gt_mm}mm p99.9={p99:.0f} gain={gain:.1f}x --")
    for nm, v in variants.items():
        if nm == "e_logp1_clip":
            continue
        p = pred(v, k=k)
        print(f"   {nm}: pred={p[0]*1000:7.1f}mm  d={(p[0]-gt_mm/1000)*1000:+7.1f}mm")
