# -*- coding: utf-8 -*-
r"""_nc_id28c.py — 重跑联合精化并保存 gonio 模型 json + 追加结论段到 nc_id28.md。"""
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
from pyFAI.goniometer import GoniometerRefinement, GeometryTransformation
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

ID28 = r"C:\Users\liu\Desktop\衍射测试图\ext1\gonio_ID28"
PX, LAM = 172e-6, 6.968e-11
GAM = {"g0": 0.0, "g17": 17.0, "g45": 45.0}
PN = ["dist", "p1_0", "p1_1", "p1_2", "p2_0", "p2_1", "p2_2",
      "r1_0", "r1_1", "r1_2", "r2_0", "r2_1", "r2_2", "r3_0", "r3_1", "r3_2"]

tf = GeometryTransformation(
    dist_expr="dist",
    poni1_expr="p1_0 + p1_1*cos(gamma) + p1_2*sin(gamma)",
    poni2_expr="p2_0 + p2_1*cos(gamma) + p2_2*sin(gamma)",
    rot1_expr="r1_0 + r1_1*cos(gamma) + r1_2*sin(gamma)",
    rot2_expr="r2_0 + r2_1*cos(gamma) + r2_2*sin(gamma)",
    rot3_expr="r3_0 + r3_1*cos(gamma) + r3_2*sin(gamma)",
    param_names=PN, pos_names=["gamma"])

posf = lambda md: (float(md),)
p0 = dict(dist=0.2847, p1_0=524.5 * PX, p1_1=0, p1_2=0,
          p2_0=510.1 * PX, p2_1=0, p2_2=0,
          r1_0=np.radians(0.34), r1_1=0, r1_2=0,
          r2_0=np.radians(0.38), r2_1=0, r2_2=0, r3_0=0, r3_1=0, r3_2=0)
det = Detector(pixel1=PX, pixel2=PX, max_shape=(1043, 981))
cal = get_calibrant("LaB6"); cal.wavelength = LAM

gr = GoniometerRefinement(p0, pos_function=posf, trans_function=tf,
                          detector=det, wavelength=LAM,
                          param_names=PN, pos_names=["gamma"])
for tag, gdeg in GAM.items():
    g = np.radians(gdeg)
    img = fabio.open(os.path.join(ID28, "det130_%s_0001p.cbf" % tag)).data
    cp = ControlPoints(os.path.join(ID28, "det130_%s_0001p.npt" % tag))
    pv = np.array([p0[k] for k in PN])
    d6 = tf(pv, (g,))
    ai0 = AzimuthalIntegrator(*d6, detector=det, wavelength=LAM)
    gr.new_geometry(tag, image=img, metadata=g, control_points=cp,
                    calibrant=cal, geometry=ai0)
gr.refine3(verbose=False)
chi2 = gr.chi2()
print("chi2 =", chi2, flush=True)

out_json = os.path.join(ROOT, "data", "real", "id28_gonio_model.json")
gr.save(out_json)
print("saved ->", out_json, flush=True)

# verify round-trip
from pyFAI.goniometer import Goniometer
g2 = Goniometer.sload(out_json)
ai = g2.get_ai(np.radians(45.0))
f2 = ai.getFit2D()
print("roundtrip g45: D=%.2fmm c=(%.1f,%.1f)" % (ai.dist * 1000, f2["centerX"], f2["centerY"]), flush=True)

mdpath = os.path.join(ROOT, "docs", "nc_id28.md")
app = ["", "## 判定结论（ID28 倾角三连 · E1f 倾角拟合的跨族外推检验）", "",
       "1. **g0 平面帧：V3 零样本第 9 套独立复验通过** —— 相鉴 LaB6✔、与 16 参数联合模型 Δc=0.7px、ΔD=-2.7mm",
       "   （与单帧 pyFAI-ref 亦 Δc=1.09px；D≈282-285mm，cbf 头 157mm/beam_xy 系未更新默认值不可用）。",
       "2. **17°/45° 大倾角帧 = 双法共域界**：V3 平面解相错（Quartz）+ rot=0 + D 掉向 D·cosγ（282→232→202mm）；",
       "   官方单帧 GR.refine3 多初值亦崩（D=35m/10.4km、中心出画布）——单帧 λ-D-rot 简并大开，非我方独有缺陷。",
       "   E1f 倾角拟合（ceria 0.1° 级）外推 2 个数量级至 17°+ 不适用，属预期域界。",
       "3. **正解 = 多帧联合 gonio 模型**（绕固定轴过样品旋转的精确 Rodrigues 参数化：",
       "   dist 共享 + poni/rot 各分量 a+b·cosγ+c·sinγ，16 参数）：refine3 收敛 chi2=0.126→3.7e-8，",
       "   dist=284.56mm 三帧不变；**帧间旋转增量 17.73°/45.21° vs 电机真值 17°/45°，轴 |cos|=1.0000**",
       "   （轴≈纯 rot1 轴，束斑沿 x 滑出画布，g45 中心画外 1150px）。",
       "4. 模型已存 data/real/id28_gonio_model.json（可复用：BM20 121 帧 / D2AM 16 帧族同款生成器）。",
       ""]
with open(mdpath, "a", encoding="utf-8") as fh:
    fh.write("\n".join(app) + "\n")
print("conclusion appended ->", mdpath, flush=True)
