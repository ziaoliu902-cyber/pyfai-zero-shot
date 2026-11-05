# -*- coding: utf-8 -*-
r"""_ceria_duel.py — MX_ceria 7 档距离扫描双臂对打（Pilatus6M, CeO2 @ 0.972386Å, D 150→450mm）。

ref 臂 = 官方 poni 直读（2013 年标定，dist 149.0~449.5mm 自洽，无需精修）；
V3 臂 = SolveWorker 零样本（λ/px 束线元数据先验）。
判定：Δc(Fit2D px) / ΔD(mm) / 相鉴（期望 CeO2，若 V3 相库无 CeO2 属覆盖问题需注明）/
D 跟踪斜率（V3 D 对官方 D 线性回归，应为 1.0）。
注：ceria_400 的 cbf 在测试图根目录，poni/npt 在 MX_ceria/。
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

BASE = r"C:\Users\liu\Desktop\衍射测试图\ext1\MX_ceria"
ROOTIMG = r"C:\Users\liu\Desktop\衍射测试图"
LAM_A = 0.972386
PX_UM = 172.0

DISTS = [150, 200, 250, 300, 350, 400, 450]

rows = []
for d in DISTS:
    tag = "D%d" % d
    cbf = os.path.join(ROOTIMG, "ceria_%d_1_0001.cbf" % d) if d == 400 else \
        os.path.join(BASE, "ceria_%d_1_0001.cbf" % d)
    poni = os.path.join(BASE, "ceria_%d_1_0001.poni" % d)
    try:
        ai_ref = pyFAI.load(poni)
        f2 = ai_ref.getFit2D()
        ref = dict(D=ai_ref.dist * 1000, c=(f2["centerX"], f2["centerY"]),
                   rot=(ai_ref.rot1, ai_ref.rot2, ai_ref.rot3))
        img = fabio.open(cbf).data.astype(np.float64)
        img = np.where(img >= 2 ** 31, 0.0, img)  # Pilatus overflow sentinel
        box = {}
        w = SolveWorker(img, LAM_A, PX_UM)
        w.done.connect(lambda res: box.update(r=res))
        w.run()
        res = box.get("r")
    except Exception as exc:
        rows.append(dict(tag=tag, ph="ERR", err=repr(exc)))
        print("%s: FRAME ERROR %r" % (tag, exc), flush=True)
        continue
    if not res:
        rows.append(dict(tag=tag, ph="FAIL", err="no result", refD=ref["D"]))
        print("%s: V3 SOLVE FAIL (ref D=%.1f)" % (tag, ref["D"]), flush=True)
        continue
    ai3 = hybrid_ai(res, PX_UM)
    f3 = ai3.getFit2D()
    dc = float(np.hypot(f3["centerX"] - ref["c"][0], f3["centerY"] - ref["c"][1]))
    dD = ai3.dist * 1000 - ref["D"]
    rows.append(dict(tag=tag, ph=res.get("ph"), dc=dc, dD=dD,
                     refD=ref["D"], v3D=ai3.dist * 1000,
                     refc=ref["c"], v3c=(f3["centerX"], f3["centerY"])))
    print("%s: ph=%-7s Δc=%7.2fpx ΔD=%+8.2fmm (ref D=%.1f V3 D=%.1f)"
          % (tag, res.get("ph"), dc, dD, ref["D"], ai3.dist * 1000), flush=True)

ok = [r for r in rows if "dc" in r]
print("\n== summary ==", flush=True)
sum_lines = []
if ok:
    dc = np.array([r["dc"] for r in ok])
    dD = np.array([r["dD"] for r in ok])
    x = np.array([r["refD"] for r in ok])
    y = np.array([r["v3D"] for r in ok])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, inter = np.linalg.lstsq(A, y, rcond=None)[0]
    s = "V3 ok %d/%d | Δc median %.1fpx (p90 %.1f) | ΔD median %+.1fmm | D跟踪 slope=%.4f intercept=%+.1fmm" % (
        len(ok), len(rows), np.median(dc), np.percentile(dc, 90), np.median(dD), slope, inter)
else:
    slope, inter = np.nan, np.nan
    s = "V3 ok 0/%d" % len(rows)
print(s, flush=True)
sum_lines.append(s)
phs = [r.get("ph") for r in rows]
pv = "phase votes: %s" % {p: phs.count(p) for p in set(phs) if p}
print(pv, flush=True)

md = ["# MX_ceria 7 档距离扫描双臂对打（Pilatus6M, CeO2@0.972386Å, D 150→450mm）", "",
      "ref = 官方 poni 直读（2013 标定，无精修）；V3 = SolveWorker 零样本（λ/px 元数据先验）。", "",
      "| 帧 | V3 相 | Δc(px) | ΔD(mm) | ref D(mm) | V3 D(mm) | ref c(px) | V3 c(px) |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "dc" in r:
        md.append("| %s | %s | %.1f | %+.1f | %.1f | %.1f | (%.0f,%.0f) | (%.0f,%.0f) |" % (
            r["tag"], r["ph"], r["dc"], r["dD"], r["refD"], r["v3D"],
            r["refc"][0], r["refc"][1], r["v3c"][0], r["v3c"][1]))
    else:
        md.append("| %s | %s | - | - | %s | - | - | - |" % (
            r["tag"], r["ph"], ("%.1f" % r["refD"]) if "refD" in r else "-"))
md += ["", "D跟踪: V3_D = %.4f * ref_D %+.1f mm" % (slope, inter) if ok else "",
       "", pv, ""]
with open(ROOT + r"\docs\nc_ceria.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")
print("md -> docs/nc_ceria.md", flush=True)
