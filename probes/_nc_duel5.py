# -*- coding: utf-8 -*-
r"""_nc_duel5.py — 经验半径锐度终裁（无相/无λ/无D，图自仲裁）。

方法：
  ① 在锚中心（基准 json 的 center_ref）做径向谱，找显著峰 → 经验环半径 r̂
  ② 各候选中心各自做径向谱，在固定 r̂±8px 窗内取（max − p10）为锐度
  ③ 最锐者胜——半径来自图本身，无相库/λ/D 依赖，无臂偏袒
对照：ceria_150 应复现"GT-Fit2D 最锐"（方法学阳性对照）。
"""
import json
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

from gui_solver import load_any

ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
BJ = json.load(open(os.path.join(ROOT, "docs", "zoo_benchmark_v3.json"),
                    encoding="utf-8"))
REFS = {r["tag"]: r["center_ref"] for r in BJ if r.get("center_ref")}
AJ = json.load(open(os.path.join(ROOT, "docs", "nc_audit_v0.json"),
                    encoding="utf-8"))
REFS.update({r["tag"]: r["cref"] for r in AJ["rows"]
             if r.get("cref")})


def profile(img, cx, cy, dr=0.25):
    rmax = min(max(img.shape) * 1.2, 4000)
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - cy, xx - cx)
    bins = np.arange(0, rmax + dr, dr)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    i, v = idx[ok], img.ravel()[ok]
    cnt = np.bincount(i, minlength=len(bins) - 1)
    p = np.bincount(i, weights=v, minlength=len(bins) - 1) / np.maximum(cnt, 1)
    return bins[:-1] + dr / 2, p


def empirical_radii(img, anchor, n_want=5):
    pr, pp = profile(img, anchor[0], anchor[1])
    r_edge = 0.9 * min(anchor[0], anchor[1], img.shape[1] - anchor[0],
                       img.shape[0] - anchor[1])
    m = (pr > 60) & (pr < r_edge)
    pr, pp = pr[m], pp[m]
    med = np.median(pp)
    mad = np.median(np.abs(pp - med))
    k = 5
    sm = np.convolve(pp, np.ones(k) / k, "same")
    peaks = []
    for j in range(2, len(sm) - 2):
        if (sm[j] >= sm[j - 1] and sm[j] >= sm[j + 1] and sm[j] >= sm[j - 2]
                and sm[j] >= sm[j + 2] and pp[j] > med + 6 * mad):
            peaks.append((pp[j], pr[j]))
    peaks.sort(reverse=True)
    sel = sorted([r for _, r in peaks[:n_want]])
    # 去重（相邻 <15px 合并取强者已隐含于排序，粗去重即可）
    dedup = []
    for r in sel:
        if not dedup or r - dedup[-1] > 15:
            dedup.append(r)
    return dedup


def sharp_at(pr, pp, r0, half=8.0):
    m = np.abs(pr - r0) < half
    if m.sum() < 5:
        return 0.0
    return float(pp[m].max() - np.percentile(pp[m], 10))


CASES = [
    ("ceria_150", os.path.join(ROOT, "data", "real", "MX_ceria",
                               "ceria_150_1_0001.cbf"),
     os.path.join(ROOT, "data", "real", "MX_ceria", "ceria_150_1_0001.poni"),
     "ceria_150"),
    ("halfccd", os.path.join(ZOO, "halfccd.edf"),
     os.path.join(ZOO, "halfccd.poni"), "halfccd"),
    ("mar3450", os.path.join(ZOO, "LaB6_260210.mar3450"),
     os.path.join(ZOO, "Mar3450.poni"), "LaB6_260210"),
]

for tag, img_f, poni_f, ref_key in CASES:
    os.chdir(os.path.dirname(poni_f))
    p = pyFAI.load(poni_f)
    naive = (p.poni2 / float(p.detector.pixel2), p.poni1 / float(p.detector.pixel1))
    f2d = p.getFit2D()
    gt2d = (f2d["centerX"], f2d["centerY"])
    anchor = REFS[ref_key]
    img = load_any(img_f)
    img = np.where(img >= 4e9, 0.0, img)
    radii = empirical_radii(img, anchor)
    print(f"== {tag} ==  锚={anchor[0]:.1f},{anchor[1]:.1f}  "
          f"经验半径={[f'{r:.0f}' for r in radii]}", flush=True)
    for ctag, c in (("GT-naive", naive), ("GT-Fit2D", gt2d),
                    ("基准ref", anchor)):
        pr, pp = profile(img, c[0], c[1])
        pk = [f"{sharp_at(pr, pp, r0):.0f}" for r0 in radii]
        tot = sum(float(x) for x in pk)
        print(f"  {ctag:10} ({c[0]:8.1f},{c[1]:8.1f})  锐度={pk}  "
              f"Σ={tot:.0f}", flush=True)
    print(flush=True)
