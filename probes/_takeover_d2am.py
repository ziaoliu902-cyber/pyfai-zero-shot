# -*- coding: utf-8 -*-
"""接手核查：D2AM(Xpad S540) 官方 npt 控制点 → 几何真值，看中心到底在哪。

V3 零样本在 D2AM 上 Δc 中位 1385px。先确认：ref 中心是否真的在画布外？
（若在画布内，V3 失败另有原因，值得修；若在画布外，就是已知的"中心出画布"域界。）
"""
import os
import re
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from pyFAI.calibrant import Calibrant
from pyFAI.geometryRefinement import GeometryRefinement

NPT = os.path.join(os.path.expanduser("~"), "Desktop", "衍射测试图", "ext1",
                   "LaB6_gonio_D2AM")
PIX = 130e-6          # Xpad S540: 130 µm
LAM = 4.9593677168e-11


def parse_npt(path):
    """返回 [(2theta, [(x,y), ...]), ...]，弧度 / 像素。"""
    groups = []
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"2theta:\s*([\d.eE+-]+)", line)
            if m:
                cur = (float(m.group(1)), [])
                groups.append(cur)
                continue
            m = re.match(r"point:\s*x=([\d.eE+-]+)\s+y=([\d.eE+-]+)", line)
            if m and cur is not None:
                cur[1].append((float(m.group(1)), float(m.group(2))))
    return groups


def npt_dspacing(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("dspacing:"):
                return np.array([float(v) for v in line.split(":", 1)[1].split()])
    return None


# pyFAI 2025.12 里 Calibrant("LaB6") 的 dspacing 为 None（数据未装载），
# 用 npt 头部自带的 dspacing 表构造标样。
cal = Calibrant()
cal.dspacing = npt_dspacing(os.path.join(NPT, "16Dec08D5_1725-rsz.npt"))
cal.wavelength = LAM

for tag in ("1725", "1742", "1791"):
    path = os.path.join(NPT, f"16Dec08D5_{tag}-rsz.npt")
    groups = parse_npt(path)
    data = []
    tth_cal = np.asarray(cal.get_2th())
    for tth, pts in groups:
        k = int(np.argmin(np.abs(tth_cal - tth)))
        for x, y in pts:
            data.append((x, y, k, 1.0))
    data = np.asarray(data, float)
    best = None
    for D0 in (0.5, 0.8, 1.2):
        gr = GeometryRefinement(data=data, calibrant=cal,
                                dist=D0, poni1=0.5 * PIX * 289,
                                poni2=0.5 * PIX * 480,
                                pixel1=PIX, pixel2=PIX,
                                wavelength=LAM, rot1=0.0, rot2=0.0, rot3=0.0)
        try:
            gr.refine3(fix=["wavelength"])
        except Exception as e:
            print(tag, D0, "refine fail:", e)
            continue
        chi = gr.chi2(tuple(gr.param)) / max(len(data), 1)
        if best is None or chi < best[0]:
            best = (chi, gr)
    if best is None:
        continue
    chi, gr = best
    cx = gr.poni2 / PIX - 0.5
    cy = gr.poni1 / PIX - 0.5
    inside = (0 <= cx < 960) and (0 <= cy < 578)
    print(f"帧{tag}: 点={len(data):4d} chi2={chi:9.4f} dist={gr.dist*1000:7.1f}mm "
          f"center=({cx:8.1f},{cy:8.1f})px rot=({np.degrees(gr.rot1):+6.2f},"
          f"{np.degrees(gr.rot2):+6.2f})° → 中心{'在画布内' if inside else '在画布外'}")
