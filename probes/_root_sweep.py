# -*- coding: utf-8 -*-
r"""_root_sweep.py — ext1 root 杂项 1:1 img+poni 对 V3 零样本清扫（ext1 收尾）。

配对清单（manifest 确认 1:1 无歧义）：
  Eiger4M.edf+Eiger4M.poni / Pilatus1M.edf+Pilatus1M.poni / Fairchild.edf+Fairchild.poni
  halfccd.edf+halfccd.poni / LaB6_0020.edf+LaB6.poni / LaB6_260210.mar3450+Mar3450.poni
  LaB6_18.57keV_frame_13.edf+LaB6_3.poni / Pilatus6M.cbf+Pilatus6M.poni / Eiger9M.h5+Eiger9M.poni
跳过：mock/moke.tif（合成单测图）、orientation（1 edf 对多 poni 配对歧义，需专门设计）。
ref = 官方 poni 直读；V3 = SolveWorker 零样本（λ 取自 poni=束线标定口径，px 取自 poni 探测器）。
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

PAIRS = [
    ("Eiger4M", "Eiger4M.edf", "Eiger4M.poni"),
    ("Pilatus1M", "Pilatus1M.edf", "Pilatus1M.poni"),
    ("Fairchild", "Fairchild.edf", "Fairchild.poni"),
    ("halfccd", "halfccd.edf", "halfccd.poni"),
    ("LaB6_0020", "LaB6_0020.edf", "LaB6.poni"),
    ("mar3450", "LaB6_260210.mar3450", "Mar3450.poni"),
    ("LaB6_18keV", "LaB6_18.57keV_frame_13.edf", "LaB6_3.poni"),
    ("Pilatus6M", "Pilatus6M.cbf", "Pilatus6M.poni"),
    ("Eiger9M", "Eiger9M.h5", "Eiger9M.poni"),
]

rows = []
for tag, imgf, ponif in PAIRS:
    ip = os.path.join(BASE, imgf)
    pp = os.path.join(BASE, ponif)
    try:
        ai_ref = pyFAI.load(pp)
        px_um = ai_ref.detector.pixel1 * 1e6
        lam_a = ai_ref.wavelength * 1e10
        f2 = ai_ref.getFit2D()
        ref = dict(D=ai_ref.dist * 1000, c=(f2["centerX"], f2["centerY"]))
        img = fabio.open(ip).data.astype(np.float64)
        if img.ndim == 3:
            img = img[0]
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

ok = [r for r in rows if "dc" in r]
print("\n== summary ==", flush=True)
if ok:
    dc = np.array([r["dc"] for r in ok])
    dD = np.array([r["dD"] for r in ok])
    s = "V3 ok %d/%d | Δc median %.1fpx p90 %.1f | ΔD median %+.2fmm max|ΔD| %.2f" % (
        len(ok), len(rows), np.median(dc), np.percentile(dc, 90),
        np.median(dD), np.abs(dD).max())
    print(s, flush=True)
phs = [r.get("ph") for r in rows]
pv = "phase: %s" % {p: phs.count(p) for p in set(phs) if p}
print(pv, flush=True)

md = ["# ext1 root 杂项清扫（9 对 img+poni，V3 零样本 vs 官方 poni 直读）", "",
      "跳过 mock/moke.tif（合成）与 orientation（1对多歧义）。λ/px 先验取自官方 poni。", "",
      "| 对 | λ(Å) | V3 相 | Δc(px) | ΔD(mm) | ref D(mm) | V3 D(mm) |", "|---|---|---|---|---|---|---|"]
for r in rows:
    if "dc" in r:
        md.append("| %s | %.4f | %s | %.1f | %+.1f | %.1f | %.1f |" % (
            r["tag"], r["lam"], r["ph"], r["dc"], r["dD"], r["refD"], r["v3D"]))
    else:
        md.append("| %s | - | %s | - | - | %s | - |" % (
            r["tag"], r["ph"], ("%.1f" % r["refD"]) if "refD" in r else "-"))
md += ["", "V3 ok %d/%d" % (len(ok), len(rows)), "", pv, ""]
with open(ROOT + r"\docs\nc_root_sweep.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")
print("md -> docs/nc_root_sweep.md", flush=True)
