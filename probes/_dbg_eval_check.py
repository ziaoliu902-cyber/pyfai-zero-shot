# -*- coding: utf-8 -*-
"""排障：为什么 tier_eval 的 dist MAE 达到米级。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, param_to_norm, norm_to_param
from torchfai.model import ResNet18Geo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for dname, ck_rel in (("train_v2", "runs/base_v2/best.pt"),
                      ("train_v3", "runs/base_v3/best.pt")):
    ds = TorchFAIDataset(os.path.join(ROOT, "data", dname), split="test")
    ck = torch.load(os.path.join(ROOT, ck_rel), map_location="cpu", weights_only=False)
    m = ResNet18Geo()
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    m.eval().cuda()
    print(f"\n===== {dname} + {ck_rel} =====")
    print("norm_consts:", {k: (np.round(v, 4) if isinstance(v, np.ndarray) else v)
                            for k, v in ds.norm_consts.items()})
    print("meta cfg dist_range:", ds.cfg.get("dist_range"),
          "poni_frac_range:", ds.cfg.get("poni_frac_range"))
    sel = ds._sel[:64]
    imgs = np.stack([np.asarray(ds._mimgs[int(g)]) for g in sel]).astype(np.float32)
    x = torch.from_numpy(np.log1p(imgs)).unsqueeze(1).cuda()
    with torch.no_grad(), torch.autocast("cuda", enabled=True):
        mu, lv = m(x)
    mu_n = mu.float().cpu().numpy()
    lv_n = lv.float().cpu().numpy()
    sig_n = np.exp(0.5 * lv_n)
    y_p = ds._mparams[sel].astype(np.float64)
    mu_p = norm_to_param(mu_n, ds.norm_consts)
    print("真值 dist 前5:", np.round(y_p[:5, 0], 4))
    print("预测 dist 前5:", np.round(mu_p[:5, 0], 4))
    print("真值 poni1 前5(mm):", np.round(y_p[:5, 1] * 1000, 3))
    print("预测 poni1 前5(mm):", np.round(mu_p[:5, 1] * 1000, 3))
    print("dist 误差中位(mm):", np.median(np.abs(mu_p[:, 0] - y_p[:, 0])) * 1000)
    print("mu_n[:,0] 前5:", np.round(mu_n[:5, 0], 3))
    y_n = param_to_norm(y_p, ds.norm_consts)
    print("y_n[:,0] 前5:", np.round(y_n[:5, 0], 3))
    print("sig_n[:,0] 前5:", np.round(sig_n[:5, 0], 4))
    print("img dtype/shape:", imgs.dtype, imgs.shape, "max:", imgs.max())
    # 无 autocast 对照
    with torch.no_grad():
        mu2, lv2 = m(x)
    print("fp32 mu_n[:,0] 前5:", np.round(mu2.float().cpu().numpy()[:5, 0], 3))
