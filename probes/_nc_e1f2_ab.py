# -*- coding: utf-8 -*-
r"""_nc_e1f2_ab.py — E1f2 A/B：倾角接线的四个几何变体全面对比。

背景：SolveWorker 对 native 帧在 5× 池化画布上拟合 tf → imp<5% →
tilt_on=False；E1f 证明原生图拟合 imp=34-39% 且全几何锐度追平 GT 62-98%。
变体（ceria_150/250/300/400，全部原生图拟合 tf，nsec=36）：
  V0 现行生产（画布 tf 门关，hybrid=平面中心+rot0/或400路径）
  V1 平面中心 + tf_native rot（只借 rot——现 hybrid_ai 规则换原生 tf）
  V2 tf 三元组（dist+poni+rot 全自洽）
  V3 solver D + tf poni/rot（交付先例）
指标：自检 dev_max（产品判词口径）｜Σ锐度｜ΔD vs GT｜中心 vs GT-Fit2D。
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from PyQt5.QtWidgets import QApplication
app = QApplication([])
import i18n
i18n.set_lang("zh")

import pyFAI
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from gui_solver import load_any, SolveWorker, hybrid_ai
from poni_out import fit_tilt
from torchfai.phases import phase_qs

PX = 172e-6
LAM = 0.972386
CERIA = os.path.join(ROOT, "data", "real", "MX_ceria")


def mk(dist, poni1, poni2, rot1, rot2):
    return AzimuthalIntegrator(dist=dist, poni1=poni1, poni2=poni2,
                               rot1=rot1, rot2=rot2, rot3=0.0,
                               wavelength=LAM * 1e-10,
                               detector=Detector(pixel1=PX, pixel2=PX))


def sigma_sharp(ai, img, qs):
    q, I = ai.integrate1d(img, 3000, unit="q_A^-1", method="nosplit_csr",
                          dummy=-1)
    tot = 0.0
    for qc in qs:
        m = np.abs(q - qc) < 0.02
        if m.sum() < 5:
            continue
        tot += float(I[m].max() - np.percentile(I[m], 10))
    return tot


def dev_max(ai, img, lam, qs):
    """自检判词口径：线位 ±0.2° 窗找峰，最大 |Δ2θ|。"""
    tth, I = ai.integrate1d(img, 16000, unit="2th_deg",
                            method="bbox,splitpixel")
    med = float(np.median(I))
    thr = med + 0.05 * (I.max() - med)
    devs = []
    for q in qs:
        t_pred = np.degrees(2 * np.arcsin(q * lam / (4 * np.pi)))
        if not (tth[0] + 0.5 < t_pred < tth[-1] - 0.5):
            continue
        win = (tth > t_pred - 0.20) & (tth < t_pred + 0.20)
        wt, wi = tth[win], I[win]
        if wt.size < 5 or wi.max() <= thr:
            continue
        k = int(np.argmax(wi))
        if k < 4 or k > len(wi) - 5:
            continue
        t_meas = wt[k]
        if 0 < k < len(wi) - 1:
            y0, y1, y2 = wi[k - 1], wi[k], wi[k + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                t_meas = wt[k] + 0.5 * (y0 - y2) / den * (wt[1] - wt[0])
        devs.append(abs(t_meas - t_pred))
    return max(devs) if devs else float("nan")


qs_all = np.asarray(phase_qs("CeO2"), float)
qs_sharp = [q for q in qs_all if 1.4 < q < 3.4][:6]

for d in (150, 250, 300, 400):
    f = os.path.join(CERIA, f"ceria_{d}_1_0001.cbf")
    img = load_any(f)
    img = np.where(img >= 4e9, 0.0, img)
    gt = pyFAI.load(f.replace(".cbf", ".poni"))
    g2 = gt.getFit2D()
    gt_D = gt.dist * 1000

    box = {}
    w = SolveWorker(img, LAM, PX * 1e6)
    w.done.connect(lambda r: box.update(r=r))
    w.run()
    res = box["r"]
    cx, cy, D_px = res["cx"], res["cy"], res["D_px"]
    s = res.get("s", 1)

    # 原生图 tf（E1f 参数）
    tfn = fit_tilt(img, cx, cy, D_px, phase_qs("CeO2"), LAM, PX, PX,
                   sfull=1, nsec=36, mass_rel=0.10, bump=1.5)

    variants = {}
    variants["V0现行"] = hybrid_ai(res, PX * 1e6)
    if tfn:
        variants["V1中心+rot"] = mk(D_px * PX, cy * PX, cx * PX,
                                    tfn["rot1"], tfn["rot2"])
        variants["V2tf三元"] = mk(tfn["dist"], tfn["poni1"], tfn["poni2"],
                                  tfn["rot1"], tfn["rot2"])
        variants["V3D解+tfP/R"] = mk(D_px * PX, tfn["poni1"], tfn["poni2"],
                                     tfn["rot1"], tfn["rot2"])
    print(f"== ceria_{d} ==  res: tilt_on={res.get('tilt_on')} "
          f"native={res.get('native')} center_src={res.get('center_src')}"
          + (f"  tfn: imp={tfn['imp']*100:.0f}% npts={tfn['npts']}"
             if tfn else "  tfn=None"), flush=True)
    for tag, ai in variants.items():
        dv = dev_max(ai, img, LAM, qs_all)
        sg = sigma_sharp(ai, img, qs_sharp)
        dd = ai.dist * 1000 - gt_D
        f2 = ai.getFit2D()
        dc = np.hypot(f2["centerX"] - g2["centerX"],
                      f2["centerY"] - g2["centerY"])
        print(f"  {tag:12} dev={dv:6.3f}°  Σ={sg:9.3e}  ΔD={dd:+7.2f}mm"
              f"  中心vsGT={dc:6.2f}px", flush=True)
    print(flush=True)
