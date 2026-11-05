# -*- coding: utf-8 -*-
"""小范围实测 pyFAI 官方标定引擎（手工流程内核）在同卷上的成绩。

两种初值口径：
  T = 真值几何起步（老师傅上限：人把机器摆位摸得很准再标）
  N = NN 粗解起步（我们的 AI 给它打下手）
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.simulator import get_detector
from torchfai.phases import PHASE_NAMES

from pyFAI.calibrant import get_calibrant
from pyFAI.goniometer import SingleGeometry
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "data/train_v2/meta.json"),
                     encoding="utf-8"))["config"]
db = get_detector(cfg["detector_name"], cfg["binning"])
from pyFAI.detectors import Detector as PyfaiDetector
pdet = PyfaiDetector(pixel1=db.pixel, pixel2=db.pixel,
                     max_shape=db.shape)      # 干净 binned 探测器（无原生 mask 干扰）
ds = TorchFAIDataset(os.path.join(ROOT, "data/train_v2"), split="test")
ck = torch.load(os.path.join(ROOT, "runs/base_v2/best.pt"),
                map_location="cpu", weights_only=False)
model = ResNet18Geo()
model.load_state_dict(ck["model"] if "model" in ck else ck)
model.eval().cuda()

PH2CAL = {"LaB6": "LaB6", "CeO2": "CeO2", "AgBh": "AgBh",
          "Al2O3": "alpha_Al2O3", "Quartz": "quartz",
          "Cristobalite": "cristobaltite"}

# 选样：每个有官方标样的相 × 近/中/远各 1
rng = np.random.default_rng(5)
picks = []
by_ph = {}
for k in rng.permutation(len(ds._sel)):
    g = int(ds._sel[k])
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    if ph not in PH2CAL:
        continue
    d = ds._mparams[g][0]
    band = "近" if d < 0.39 else ("中" if d < 0.77 else "远")
    by_ph.setdefault(ph, set())
    if band not in by_ph[ph]:
        by_ph[ph].add(band)
        picks.append((k, ph, band))
    if len(picks) >= 12:
        break

print(f"{'样本':>4s} {'相':8s}{'段':3s} {'初值':4s} {'耗时':>5s} {'控制点':>5s} "
      f"{'dist误差':>9s} {'poni1误差':>9s} {'rot1误差':>8s}")
for k, ph, band in picks:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g]).astype(np.float32)
    tp = ds._mparams[g].astype(np.float64)
    cal = get_calibrant(PH2CAL[ph])
    cal.wavelength = cfg["wavelength"]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img))[None, None].cuda()
        mu, _ = model(x)
        pred = norm_to_param(mu[0].cpu().numpy().astype(np.float64), ds.norm_consts)
    for tag, init in (("T", tp.copy()), ("N", pred.copy())):
        ai0 = AzimuthalIntegrator(dist=init[0], poni1=init[1], poni2=init[2],
                                  rot1=init[3], rot2=init[4], rot3=init[5],
                                  detector=pdet, wavelength=cfg["wavelength"])
        t0 = time.time()
        try:
            sg = SingleGeometry(f"sample{k}", image=img, calibrant=cal,
                                detector=pdet, geometry=ai0)
            sg.extract_cp(max_rings=12)
            ncp = len(sg.control_points.get_labels())
            gr = getattr(sg, "geometry_refinement", None)
            if gr is None:
                from pyFAI.geometryRefinement import GeometryRefinement
                gr = GeometryRefinement(sg.control_points.getList(),
                                        detector=pdet, calibrant=cal,
                                        dist=ai0.dist, poni1=ai0.poni1,
                                        poni2=ai0.poni2, rot1=ai0.rot1,
                                        rot2=ai0.rot2, rot3=ai0.rot3,
                                        wavelength=cfg["wavelength"])
            gr.refine3(fix=["rot1", "rot2", "rot3"])  # 官方手册口径:转角多靠机械精度
            gp = gr.getPyFAI()
            e = np.array([gp["dist"] - tp[0], gp["poni1"] - tp[1],
                          gp["poni2"] - tp[2]]) * 1000
            print(f"{k:4d} {ph:8s}{band:3s} {tag:4s} {time.time()-t0:4.1f}s "
                  f"{ncp:5d} {e[0]:+9.2f} {e[1]:+9.2f}     (转角已固定)")
        except Exception as ex:
            print(f"{k:4d} {ph:8s}{band:3s} {tag:4s}  失败: {str(ex)[:60]}")
