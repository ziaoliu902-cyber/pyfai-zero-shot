# -*- coding: utf-8 -*-
r"""_d2am_duel.py — D2AM 16 帧逐帧双臂对打（Xpad_S540_flat, LaB6 @ 0.4959A）。

ref 臂：官方 npt 控制点 -> GeometryRefinement.refine3（初值扫描 D{0.5,0.8,1.2} 取 chi2 最小）
V3 臂：SolveWorker 零样本（lam=0.4959A, px=130um——npt 波长属合法先验口径? 见注：
       与 zoo 口径一致，λ 来自束线元数据而非 GT 拟合）。
判定：Δc(Fit2D 口径 px) / ΔD(mm) / V3 相鉴。
"""
import glob
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

import fabio
from pyFAI.calibrant import get_calibrant
from pyFAI.control_points import ControlPoints
from pyFAI.detectors import detector_factory
from pyFAI.geometryRefinement import GeometryRefinement

from gui_solver import load_any, SolveWorker, hybrid_ai

BASE = r"C:\Users\liu\Desktop\衍射测试图\ext1\LaB6_gonio_D2AM"
LAM_M = 4.9593677168e-11
LAM_A = LAM_M * 1e10
PX_UM = 130.0
det = detector_factory("Xpad_S540_flat")
print("detector:", det.name, det.shape, det.pixel1)

cal = get_calibrant("LaB6")
cal.wavelength = LAM_M

edfs = sorted(glob.glob(os.path.join(BASE, "16Dec08D5_*-rsz.edf")))
print("frames:", len(edfs))

def ref_solve(npt_path):
    cp = ControlPoints(npt_path)
    data = np.array(cp.getList(), dtype=np.float64)
    best, bchi = None, np.inf
    h, w = det.shape
    for D0 in (0.5, 0.8, 1.2):
        gr = GeometryRefinement(data=data, calibrant=cal, dist=D0,
                                poni1=h / 2 * det.pixel1, poni2=w / 2 * det.pixel2,
                                rot1=0, rot2=0, rot3=0, detector=det, wavelength=LAM_M)
        try:
            # λ 是束线元数据(0.4959Å)，必须冻结——不冻则 D-λ 简并漂移致环角越界崩
            gr.refine3(fix=["wavelength"])
            c = gr.chi2()
        except Exception as exc:
            print("  ref skip D0=%.1f: %r" % (D0, exc), flush=True)
            continue
        if c < bchi:
            best, bchi = gr, c
    if best is None:
        raise RuntimeError("all D0 failed for %s" % npt_path)
    f2 = best.getFit2D()
    return dict(D=best.dist, c=(f2["centerX"], f2["centerY"]),
                rot=(best.rot1, best.rot2, best.rot3), chi2=bchi)

rows = []
for e in edfs:
    tag = os.path.basename(e)[10:14]
    try:
        r = ref_solve(e[:-4] + ".npt")
        img = load_any(e)
        img = np.where(img >= 4e9, 0.0, img)
        box = {}
        w = SolveWorker(img, LAM_A, PX_UM)
        w.done.connect(lambda res: box.update(r=res))
        w.run()
        res = box.get("r")
    except Exception as exc:
        rows.append((tag, "ERR", np.nan, np.nan, np.nan, np.nan))
        print("%s: FRAME ERROR %r" % (tag, exc), flush=True)
        continue
    if res:
        ai3 = hybrid_ai(res, PX_UM)
        f3 = ai3.getFit2D()
        dc = float(np.hypot(f3["centerX"] - r["c"][0], f3["centerY"] - r["c"][1]))
        dD = ai3.dist * 1000 - r["D"] * 1000
        rows.append((tag, res.get("ph"), dc, dD, r["D"] * 1000, ai3.dist * 1000))
        print("%s: ph=%-7s Δc=%7.2fpx ΔD=%+8.2fmm (ref D=%.1f V3 D=%.1f)"
              % (tag, res.get("ph"), dc, dD, r["D"] * 1000, ai3.dist * 1000), flush=True)
    else:
        rows.append((tag, "FAIL", np.nan, np.nan, r["D"] * 1000, np.nan))
        print("%s: V3 SOLVE FAIL (ref D=%.1f)" % (tag, r["D"] * 1000), flush=True)

dc = np.array([x[2] for x in rows], dtype=float)
dD = np.array([x[3] for x in rows], dtype=float)
ok = np.isfinite(dc)
print("\n== summary ==")
print("V3 ok: %d/%d | Δc median %.1fpx (p90 %.1f) | ΔD median %+.1fmm"
      % (ok.sum(), len(rows), np.nanmedian(dc), np.nanpercentile(dc, 90), np.nanmedian(dD)))

md = ["# D2AM 16 帧双臂对打（Xpad_S540_flat, LaB6@0.4959Å, 2018-01-30 D5）", "",
      "ref = 官方 npt 控制 points + GR.refine3（初值扫描）；V3 = SolveWorker 零样本（λ/px 元数据先验）。", "",
      "| 帧 | V3 相 | Δc(px) | ΔD(mm) | ref D(mm) | V3 D(mm) |", "|---|---|---|---|---|---|"]
for t, ph, a, b, c, d in rows:
    md.append("| %s | %s | %.1f | %+.1f | %.1f | %.1f |" % (t, ph, a, b, c, d))
md += ["", "V3 ok %d/%d；Δc 中位 %.1fpx；ΔD 中位 %+.1fmm" % (ok.sum(), len(rows), np.nanmedian(dc), np.nanmedian(dD))]
with open(os.path.join(ROOT, "docs", "d2am_duel.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")
print("md -> docs/d2am_duel.md")
