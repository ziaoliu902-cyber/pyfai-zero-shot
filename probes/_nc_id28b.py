# -*- coding: utf-8 -*-
r"""_nc_id28b.py — ID28 三连·正统多帧联合 gonio 标定参考臂。

模型（精确物理，绕过样品固定轴旋转 gamma）：
  dist = D（不变）
  poni1/2, rot1/2/3 = a + b*cos(gamma) + c*sin(gamma)   [Rodrigues 在固定轴下的精确形式]
参数 16 个：dist, p1_0..2, p2_0..2, r1_0..2, r2_0..2, r3_0..2
pos：gamma = 0 / 17 / 45 deg（电机真值，来自文件名）

检验（模型自由、不强制增量=步进）：
  联合收敛后 get_ai(gamma) 的帧间旋转增量是否回到 17/45 deg（模型忠实度=物理抓没抓住）。
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio
from pyFAI.calibrant import get_calibrant
from pyFAI.control_points import ControlPoints
from pyFAI.detectors import Detector
from pyFAI.goniometer import ExtendedTransformation, GoniometerRefinement

ID28 = r"C:\Users\liu\Desktop\衍射测试图\ext1\gonio_ID28"
PX = 172e-6
LAM = 6.968e-11
GAM = {"g0": 0.0, "g17": 17.0, "g45": 45.0}   # deg, motor truth

def rmat(r1, r2, r3):
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
    for i in range(3):
        if abs(w[i].real - 1) < 1e-6 and abs(w[i].imag) < 1e-6:
            a = v[:, i].real
            return th, a / np.linalg.norm(a)
    return th, None

PN = ["dist", "p1_0", "p1_1", "p1_2", "p2_0", "p2_1", "p2_2",
      "r1_0", "r1_1", "r1_2", "r2_0", "r2_1", "r2_2", "r3_0", "r3_1", "r3_2"]

from pyFAI.goniometer import GeometryTransformation

# callable object with .codes: exact Rodrigues form for rotation about a fixed
# axis through the sample  ->  poni/rot Euler components = a + b*cos(g) + c*sin(g)
trans_function = GeometryTransformation(
    dist_expr="dist",
    poni1_expr="p1_0 + p1_1*cos(gamma) + p1_2*sin(gamma)",
    poni2_expr="p2_0 + p2_1*cos(gamma) + p2_2*sin(gamma)",
    rot1_expr="r1_0 + r1_1*cos(gamma) + r1_2*sin(gamma)",
    rot2_expr="r2_0 + r2_1*cos(gamma) + r2_2*sin(gamma)",
    rot3_expr="r3_0 + r3_1*cos(gamma) + r3_2*sin(gamma)",
    param_names=PN, pos_names=["gamma"])

def pos_function(metadata):
    return (float(metadata),)   # metadata IS gamma in rad

# init: g0 anchor (from single-frame GR in _nc_id28.py run A/B: D~0.2847, c~(510,524.5), rot~(0.34,0.38,0)deg)
p0 = dict(dist=0.2847,
          p1_0=524.5 * PX, p1_1=0.0, p1_2=0.0,
          p2_0=510.1 * PX, p2_1=0.0, p2_2=0.0,
          r1_0=np.radians(0.34), r1_1=0.0, r1_2=0.0,
          r2_0=np.radians(0.38), r2_1=0.0, r2_2=0.0,
          r3_0=0.0, r3_1=0.0, r3_2=0.0)

det = Detector(pixel1=PX, pixel2=PX, max_shape=(1043, 981))
cal = get_calibrant("LaB6")
cal.wavelength = LAM

gr = GoniometerRefinement(p0, pos_function=pos_function,
                          trans_function=trans_function,
                          detector=det, wavelength=LAM,
                          param_names=PN, pos_names=["gamma"])

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

for tag, gdeg in GAM.items():
    g = np.radians(gdeg)
    img = fabio.open(os.path.join(ID28, "det130_%s_0001p.cbf" % tag)).data
    cp = ControlPoints(os.path.join(ID28, "det130_%s_0001p.npt" % tag))
    pv = np.array([p0[k] for k in PN])
    d6 = trans_function(pv, pos_function(g))       # model-generated initial geometry (6-tuple)
    ai0 = AzimuthalIntegrator(d6[0], d6[1], d6[2], d6[3], d6[4], d6[5],
                              detector=det, wavelength=LAM)
    gr.new_geometry(tag, image=img, metadata=g, control_points=cp,
                    calibrant=cal, geometry=ai0)
    print("added %s gamma=%gdeg ncp=%d" % (tag, gdeg, len(cp.getList())), flush=True)

print("\nrefining (refine3, 16 params x 3 frames)...", flush=True)
gr.refine3()
print("chi2 =", gr.chi2(), flush=True)
print("params:", {k: float(v) for k, v in zip(PN, gr.param)}, flush=True)

print("\n== per-frame geometry from joint model ==", flush=True)
sol = {}
for tag, gdeg in GAM.items():
    ai = gr.get_ai(np.radians(gdeg))
    f2 = ai.getFit2D()
    sol[tag] = dict(rot=(ai.rot1, ai.rot2, ai.rot3), D=ai.dist,
                    c=(f2["centerX"], f2["centerY"]))
    print("%s: D=%.2fmm c=(%.1f,%.1f) rot=(%+.3f,%+.3f,%+.3f)deg"
          % (tag, ai.dist * 1000, f2["centerX"], f2["centerY"],
             np.degrees(ai.rot1), np.degrees(ai.rot2), np.degrees(ai.rot3)), flush=True)

print("\n== inter-frame increments vs motor truth (17 / 45 deg, same axis) ==", flush=True)
lines = []
axes = {}
for tag, truth in (("g17", 17.0), ("g45", 45.0)):
    Rd = rmat(*sol[tag]["rot"]) @ rmat(*sol["g0"]["rot"]).T
    th, ax = angle_axis(Rd)
    axes[tag] = ax
    lines.append("g0->%s: model %.2fdeg (truth %gdeg) axis=%s"
                 % (tag, np.degrees(th), truth, np.round(ax, 3) if ax is not None else None))
a17, a45 = axes.get("g17"), axes.get("g45")
if a17 is not None and a45 is not None:
    lines.append("axis(17)-axis(45) |cos| = %.4f" % abs(float(np.dot(a17, a45))))
print("\n".join(lines), flush=True)

# also compare with V3 zero-shot from _nc_id28.py run
V3 = {"g0": dict(D=0.28182, c=(509.2, 523.8)),
      "g17": dict(D=0.23248, c=(495.4, 532.7)),
      "g45": dict(D=0.20233, c=(489.4, 518.9))}
print("\n== V3(planar) vs joint-model ==", flush=True)
for tag in GAM:
    d = sol[tag]
    dc = float(np.hypot(V3[tag]["c"][0] - d["c"][0], V3[tag]["c"][1] - d["c"][1]))
    print("%s: Δc=%.1fpx ΔD=%+.2fmm (V3 D=%.1f)" % (tag, dc, (V3[tag]["D"] - d["D"]) * 1000, V3[tag]["D"] * 1000), flush=True)

# dump md appendix
mdpath = os.path.join(ROOT, "docs", "nc_id28.md")
app = ["", "## 多帧联合 gonio 模型（ExtendedTransformation cos/sin 精确参数化, refine3）", "",
       "chi2=%g；param=%s" % (gr.chi2(), {k: round(float(v), 6) for k, v in zip(PN, gr.param)}), ""]
app += ["| 帧 | D(mm) | center | rot(deg) |", "|---|---|---|---|"]
for tag in GAM:
    d = sol[tag]
    app.append("| %s | %.2f | (%.1f,%.1f) | (%+.3f,%+.3f,%+.3f) |"
               % (tag, d["D"] * 1000, d["c"][0], d["c"][1],
                  np.degrees(d["rot"][0]), np.degrees(d["rot"][1]), np.degrees(d["rot"][2])))
app += ["", "增量检验:"] + ["- " + l for l in lines]
with open(mdpath, "a", encoding="utf-8") as fh:
    fh.write("\n".join(app) + "\n")
print("\nappended -> docs/nc_id28.md", flush=True)
