# -*- coding: utf-8 -*-
r"""_root_sweep3.py — halfccd/LaB6_0020 补跑（spline 探测器 pixel1=None → 从 spline 头读像素）。

halfccd.spline 头：valid 2048x1025, px X=46.84483µm Y=48.42252µm（各向异性 3%，先验取均值 47.63µm）。
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

import fabio
import pyFAI

from gui_solver import SolveWorker, hybrid_ai

BASE = r"C:\Users\liu\Desktop\衍射测试图\ext1\root"
os.chdir(BASE)

spl = os.path.join(BASE, "halfccd.spline")
txt = open(spl).read().splitlines()
i = txt.index("  GRID SPACING, X-PIXEL SIZE, Y-PIXEL SIZE") + 1
sx, sy = (float(v) for v in txt[i].split()[1:3])
PX_UM = (sx + sy) / 2
print("spline px: X=%.4f Y=%.4f -> prior %.4f um" % (sx, sy, PX_UM), flush=True)

PAIRS = [
    ("halfccd", "halfccd.edf", "halfccd.poni"),
    ("LaB6_0020", "LaB6_0020.edf", "LaB6.poni"),
]

rows = []
for tag, imgf, ponif in PAIRS:
    ip = os.path.join(BASE, imgf)
    pp = os.path.join(BASE, ponif)
    try:
        ai_ref = pyFAI.load(pp)
        lam_a = ai_ref.wavelength * 1e10
        f2 = ai_ref.getFit2D()
        ref = dict(D=ai_ref.dist * 1000, c=(f2["centerX"], f2["centerY"]))
        img = fabio.open(ip).data.astype(np.float64)
        img = np.where(img >= 2 ** 31, 0.0, img)
        box = {}
        w = SolveWorker(img, lam_a, PX_UM)
        w.done.connect(lambda res: box.update(r=res))
        w.run()
        res = box.get("r")
    except Exception as exc:
        rows.append(dict(tag=tag, ph="ERR", err=repr(exc)[:90]))
        print("%-11s: ERROR %r" % (tag, exc), flush=True)
        continue
    if not res:
        rows.append(dict(tag=tag, ph="FAIL", refD=ref["D"]))
        print("%-11s: V3 FAIL (ref D=%.1f)" % (tag, ref["D"]), flush=True)
        continue
    ai3 = hybrid_ai(res, PX_UM)
    f3 = ai3.getFit2D()
    dc = float(np.hypot(f3["centerX"] - ref["c"][0], f3["centerY"] - ref["c"][1]))
    dD = ai3.dist * 1000 - ref["D"]
    rows.append(dict(tag=tag, ph=res.get("ph"), dc=dc, dD=dD, refD=ref["D"],
                     v3D=ai3.dist * 1000, lam=lam_a))
    print("%-11s: ph=%-7s λ=%.4fÅ Δc=%8.2fpx ΔD=%+9.2fmm (ref D=%8.1f V3 D=%8.1f)"
          % (tag, res.get("ph"), lam_a, dc, dD, ref["D"], ai3.dist * 1000), flush=True)

extra = ["", "## 补跑2（spline px 先验 %.2fµm 均值，各向异性3%已注明）" % PX_UM, "",
         "| 对 | λ(Å) | V3 相 | Δc(px) | ΔD(mm) | ref D(mm) | V3 D(mm) |", "|---|---|---|---|---|---|---|"]
for r in rows:
    if "dc" in r:
        extra.append("| %s | %.4f | %s | %.1f | %+.1f | %.1f | %.1f |" % (
            r["tag"], r["lam"], r["ph"], r["dc"], r["dD"], r["refD"], r["v3D"]))
    else:
        extra.append("| %s | - | %s | - | - | %s | - |" % (
            r["tag"], r["ph"], ("%.1f" % r["refD"]) if "refD" in r else "-"))
with open(ROOT + r"\docs\nc_root_sweep.md", "a", encoding="utf-8") as fh:
    fh.write("\n".join(extra) + "\n")
print("appended -> docs/nc_root_sweep.md", flush=True)
