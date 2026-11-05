# -*- coding: utf-8 -*-
r"""_root_sweep2.py — root 清扫补跑 3 对（修复路径/spline 解析）。

修复：Eiger4M.edf 在测试图顶层（非 ext1\root）；halfccd.spline 需 CWD=ext1\root 才被 pyFAI 找到。
剔除：LaB6_18keV 行（LaB6_3.poni 配对为猜测，λ=1.0Å 与 18.57keV→0.668Å 矛盾，无真值意义）。
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
TOP = r"C:\Users\liu\Desktop\衍射测试图"
os.chdir(BASE)  # spline 相对路径解析

PAIRS = [
    ("Eiger4M", os.path.join(TOP, "Eiger4M.edf"), "Eiger4M.poni"),
    ("halfccd", os.path.join(BASE, "halfccd.edf"), "halfccd.poni"),
    ("LaB6_0020", os.path.join(BASE, "LaB6_0020.edf"), "LaB6.poni"),
]

rows = []
for tag, ip, ponif in PAIRS:
    pp = os.path.join(BASE, ponif)
    try:
        ai_ref = pyFAI.load(pp)
        px_um = ai_ref.detector.pixel1 * 1e6
        lam_a = ai_ref.wavelength * 1e10
        f2 = ai_ref.getFit2D()
        ref = dict(D=ai_ref.dist * 1000, c=(f2["centerX"], f2["centerY"]))
        img = fabio.open(ip).data.astype(np.float64)
        img = np.where(img >= 2 ** 31, 0.0, img)
        box = {}
        w = SolveWorker(img, lam_a, px_um)
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
    ai3 = hybrid_ai(res, px_um)
    f3 = ai3.getFit2D()
    dc = float(np.hypot(f3["centerX"] - ref["c"][0], f3["centerY"] - ref["c"][1]))
    dD = ai3.dist * 1000 - ref["D"]
    rows.append(dict(tag=tag, ph=res.get("ph"), dc=dc, dD=dD, refD=ref["D"],
                     v3D=ai3.dist * 1000, lam=lam_a))
    print("%-11s: ph=%-7s λ=%.4fÅ Δc=%8.2fpx ΔD=%+9.2fmm (ref D=%8.1f V3 D=%8.1f)"
          % (tag, res.get("ph"), lam_a, dc, dD, ref["D"], ai3.dist * 1000), flush=True)

phs = [r.get("ph") for r in rows]
print("phase: %s" % {p: phs.count(p) for p in set(phs) if p}, flush=True)

# 追加进 nc_root_sweep.md
mdf = ROOT + r"\docs\nc_root_sweep.md"
extra = ["", "## 补跑（修复路径/spline，剔除 LaB6_18keV 无真值行）", "",
         "| 对 | λ(Å) | V3 相 | Δc(px) | ΔD(mm) | ref D(mm) | V3 D(mm) |", "|---|---|---|---|---|---|---|"]
for r in rows:
    if "dc" in r:
        extra.append("| %s | %.4f | %s | %.1f | %+.1f | %.1f | %.1f |" % (
            r["tag"], r["lam"], r["ph"], r["dc"], r["dD"], r["refD"], r["v3D"]))
    else:
        extra.append("| %s | - | %s | - | - | %s | - |" % (
            r["tag"], r["ph"], ("%.1f" % r["refD"]) if "refD" in r else "-"))
with open(mdf, "a", encoding="utf-8") as fh:
    fh.write("\n".join(extra) + "\n")
print("appended -> docs/nc_root_sweep.md", flush=True)
