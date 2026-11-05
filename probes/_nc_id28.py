# -*- coding: utf-8 -*-
r"""_nc_id28.py — ID28 倾角三连独立验证（E1f 倾角拟合的跨族检验）。

数据：det130 g0/g17/g45（PILATUS 2M ROI 1043x981, 172um, LaB6, lambda_npt=0.6968A）。
真值哲学：
  1) 帧间旋转增量 = 测角仪步进（+17deg/+45deg 绕同一轴）——自洽硬真值，零外部依赖。
  2) npt 控制点 + pyFAI 官方 GeometryRefinement(refine3) = pyFAI-ref 参考几何（对照组，非 GT）。
  3) cbf 头工业侧粗几何（D=0.157m, beam_xy=(542.41,732.4)px）只做粗对照。

臂：
  V3  = 生产零样本 SolveWorker（lambda=0.7A cbf 头名义, px=172um；不偷看 npt）+ hybrid_ai。
  ref = GR(refine3)，g0 初值 rot=0；g17/g45 多初值 {rot1,rot2}x{+,-}x{17,45}deg 取 chi2 最小
        （官方工具的公平用法：人手给初值的标准流程）。

判定：
  A. 帧间增量角 theta(V3) / theta(ref) vs 17/45 deg；增量旋转轴一致性 |cos|。
  B. V3 vs ref 逐帧 Δcenter(Fit2D 口径 px) / ΔD(mm)。
输出：stdout 表 + D:\pyfai-project\docs\nc_id28.md
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
from pyFAI.calibrant import get_calibrant
from pyFAI.control_points import ControlPoints
from pyFAI.geometryRefinement import GeometryRefinement

from gui_solver import load_any, SolveWorker, hybrid_ai

ID28 = r"C:\Users\liu\Desktop\衍射测试图\ext1\gonio_ID28"
FRAMES = ["g0", "g17", "g45"]
PX = 172e-6          # m
PX_UM = 172.0        # um  (SolveWorker 口径)
LAM0 = 0.7           # A   (cbf 头名义, V3 合法先验)
LAM_REF = 6.968e-11  # m   (npt, ref 臂)
D0, BX, BY = 0.157, 542.41, 732.4   # cbf 头工业侧

def rmat(r1, r2, r3):
    """pyFAI 约定 R = Rz(rot3) @ Ry(rot2) @ Rx(rot1)。自洽口径即可。"""
    sx, cx = np.sin(r1), np.cos(r1)
    sy, cy = np.sin(r2), np.cos(r2)
    sz, cz = np.sin(r3), np.cos(r3)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def angle_axis(R):
    tr = np.clip(np.trace(R), -1.0, 3.0)
    th = float(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    w, v = np.linalg.eig(R)
    ax = None
    for i in range(3):
        if abs(w[i].real - 1) < 1e-6 and abs(w[i].imag) < 1e-6:
            a = v[:, i].real
            ax = a / np.linalg.norm(a)
            break
    return th, ax

# ---------------- A. V3 零样本 ----------------
print("== A. V3 zero-shot (SolveWorker, lam=0.7A cbf-header, px=172um) ==", flush=True)
V3 = {}
for f in FRAMES:
    img_f = os.path.join(ID28, "det130_%s_0001p.cbf" % f)
    img = load_any(img_f)
    img = np.where(img >= 4e9, 0.0, img)
    box = {}
    w = SolveWorker(img, LAM0, PX_UM)
    w.done.connect(lambda r: box.update(r=r))
    w.run()
    res = box.get("r")
    if not res:
        print(f"{f}: SOLVE FAIL", flush=True)
        V3[f] = None
        continue
    ai3 = hybrid_ai(res, PX_UM)
    f3 = ai3.getFit2D()
    V3[f] = dict(rot=(ai3.rot1, ai3.rot2, ai3.rot3), D=ai3.dist,
                 c=(f3["centerX"], f3["centerY"]),
                 ph=res.get("ph"), tf=res.get("tf"), native=res.get("native"))
    tf = res.get("tf")
    imp = ("%.0f%%" % (tf["imp"] * 100)) if tf else "-"
    print(f"{f}: ph={res.get('ph')} native={res.get('native')} imp={imp} "
          f"D={ai3.dist*1000:.2f}mm c=({f3['centerX']:.1f},{f3['centerY']:.1f}) "
          f"rot=({np.degrees(ai3.rot1):+.2f},{np.degrees(ai3.rot2):+.2f},{np.degrees(ai3.rot3):+.2f})deg",
          flush=True)

# ---------------- B. pyFAI-ref（官方 GR, 多初值） ----------------
print("\n== B. pyFAI-ref (GR.refine3, multi-init for tilt frames) ==", flush=True)
cal = get_calibrant("LaB6")
cal.wavelength = LAM_REF

def refine_frame(tag, inits):
    cp = ControlPoints(os.path.join(ID28, "det130_%s_0001p.npt" % tag))
    data = np.array(cp.getList(), dtype=np.float64)
    best, best_chi = None, np.inf
    for r0 in inits:
        gr = GeometryRefinement(data=data, calibrant=cal, dist=D0,
                                poni1=BY * PX, poni2=BX * PX,
                                rot1=r0[0], rot2=r0[1], rot3=0.0,
                                pixel1=PX, pixel2=PX, wavelength=LAM_REF)
        gr.refine3()
        chi = gr.chi2()
        if chi < best_chi:
            best, best_chi = gr, chi
    f2 = best.getFit2D()
    return dict(rot=(best.rot1, best.rot2, best.rot3), D=best.dist,
                c=(f2["centerX"], f2["centerY"]), chi2=best_chi)

REF = {}
REF["g0"] = refine_frame("g0", [(0.0, 0.0)])
d17, d45 = np.radians(17.0), np.radians(45.0)
REF["g17"] = refine_frame("g17", [(+d17, 0), (-d17, 0), (0, +d17), (0, -d17)])
REF["g45"] = refine_frame("g45", [(+d45, 0), (-d45, 0), (0, +d45), (0, -d45)])
for f in FRAMES:
    r = REF[f]
    print(f"{f}: chi2={r['chi2']:.4g} D={r['D']*1000:.2f}mm c=({r['c'][0]:.1f},{r['c'][1]:.1f}) "
          f"rot=({np.degrees(r['rot'][0]):+.2f},{np.degrees(r['rot'][1]):+.2f},{np.degrees(r['rot'][2]):+.2f})deg",
          flush=True)

# ---------------- C. 判定 ----------------
print("\n== C. verdicts ==", flush=True)
rows = []
for f in FRAMES:
    v, r = V3[f], REF[f]
    dc = float(np.hypot(v["c"][0] - r["c"][0], v["c"][1] - r["c"][1])) if v else float("nan")
    dD = (v["D"] - r["D"]) * 1000 if v else float("nan")
    dc_ind = float(np.hypot(v["c"][0] - BX, v["c"][1] - BY))
    dr_ind = float(np.hypot(r["c"][0] - BX, r["c"][1] - BY))
    rows.append((f, dc, dD, dc_ind, dr_ind))
    print(f"{f}: V3-vs-ref Δc={dc:.2f}px ΔD={dD:+.2f}mm | vs工业头 V3 {dc_ind:.1f}px ref {dr_ind:.1f}px "
          f"| ref D vs 157mm: {r['D']*1000-157:+.1f}mm", flush=True)

# inter-frame rotation increments
print("\n-- inter-frame rotation increments (hard truth: +17deg / +45deg same axis) --", flush=True)
inc_rows = []
for f, th_true in (("g17", 17.0), ("g45", 45.0)):
    line = f"g0->{f} (true {th_true:.0f}deg): "
    axes = {}
    for name, src in (("V3", V3), ("ref", REF)):
        if src[f] is None or src["g0"] is None:
            line += f"{name}: FAIL  "
            continue
        Rd = rmat(*src[f]["rot"]) @ rmat(*src["g0"]["rot"]).T
        th, ax = angle_axis(Rd)
        axes[name] = ax
        line += f"{name}: {np.degrees(th):.2f}deg  "
    if "V3" in axes and "ref" in axes and axes["V3"] is not None and axes["ref"] is not None:
        cd = abs(float(np.dot(axes["V3"], axes["ref"])))
        line += f"| axis |cos| = {cd:.3f}"
    print(line, flush=True)
    inc_rows.append(line)

# axis consistency across 17 and 45 within each arm
line = "axis(17)-axis(45) consistency: "
for name, src in (("V3", V3), ("ref", REF)):
    try:
        a1 = angle_axis(rmat(*src["g17"]["rot"]) @ rmat(*src["g0"]["rot"]).T)[1]
        a2 = angle_axis(rmat(*src["g45"]["rot"]) @ rmat(*src["g0"]["rot"]).T)[1]
        line += f"{name} |cos|={abs(float(np.dot(a1, a2))):.3f}  "
    except Exception as e:
        line += f"{name}: ERR({e})  "
print(line, flush=True)

# ---------------- md dump ----------------
md = ["# ID28 倾角三连独立验证（E1f 倾角拟合跨族检验）", "",
      "生成：_nc_id28.py ｜ 数据：det130 g0/g17/g45（PILATUS 2M ROI, LaB6, λ_npt=0.6968Å）",
      "真值哲学：帧间旋转增量=测角仪步进（+17°/+45° 同轴）自洽硬真值；ref=pyFAI GR.refine3 多初值（对照组）。", "",
      "| 臂 | 帧 | D(mm) | center | rot(deg) | Δc vs ref(px) | ΔD vs ref(mm) |",
      "|---|---|---|---|---|---|---|"]
for f in FRAMES:
    v, r = V3[f], REF[f]
    if v:
        md.append(f"| V3 | {f} | {v['D']*1000:.2f} | ({v['c'][0]:.1f},{v['c'][1]:.1f}) | "
                  f"({np.degrees(v['rot'][0]):+.2f},{np.degrees(v['rot'][1]):+.2f},{np.degrees(v['rot'][2]):+.2f}) | "
                  f"{rows[FRAMES.index(f)][1]:.2f} | {rows[FRAMES.index(f)][2]:+.2f} |")
    md.append(f"| ref | {f} | {r['D']*1000:.2f} | ({r['c'][0]:.1f},{r['c'][1]:.1f}) | "
              f"({np.degrees(r['rot'][0]):+.2f},{np.degrees(r['rot'][1]):+.2f},{np.degrees(r['rot'][2]):+.2f}) | — | — |")
md += ["", "## 帧间旋转增量（硬真值 17°/45°）"] + ["- " + l for l in inc_rows]
with open(os.path.join(ROOT, "docs", "nc_id28.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")
print("\nmd -> docs/nc_id28.md", flush=True)
