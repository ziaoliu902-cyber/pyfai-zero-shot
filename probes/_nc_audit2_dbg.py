# -*- coding: utf-8 -*-
"""_nc_audit2_dbg.py — NMS 修复离线单测（ceria_150/E9M，不 import 审计模块）。"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
from gui_solver import load_any


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


def radii_nms(img, anchor, n_want=4):
    pr, pp = profile(img, anchor[0], anchor[1])
    r_edge = 0.9 * min(anchor[0], anchor[1], img.shape[1] - anchor[0],
                       img.shape[0] - anchor[1])
    m = (pr > 60) & (pr < r_edge)
    if m.sum() < 50:
        return []
    pr, pp = pr[m], pp[m]
    med = np.median(pp)
    mad = max(np.median(np.abs(pp - med)), 1e-9)
    for k in (8, 6, 4, 3, 2):
        thr = med + k * mad
        sm = np.convolve(pp, np.ones(5) / 5, "same")
        cand = []
        for j in range(2, len(sm) - 2):
            if not (60 < pr[j] < r_edge):
                continue
            if (sm[j] >= sm[j - 1] and sm[j] >= sm[j + 1]
                    and sm[j] >= sm[j - 2] and sm[j] >= sm[j + 2]
                    and pp[j] > thr):
                cand.append((pp[j], pr[j]))
        cand.sort(reverse=True)
        sel = []
        for h, r in cand:
            if all(abs(r - r0) > 25 for r0 in sel):
                sel.append(r)
            if len(sel) >= n_want:
                break
        dedup = sorted(sel)
        if len(dedup) >= 3:
            return dedup
    return []


img = load_any(os.path.join(ROOT, "data", "real", "MX_ceria",
                            "ceria_150_1_0001.cbf"))
img = np.where(img >= 4e9, 0.0, img)
print("ceria_150 NMS radii:", radii_nms(img, (1227.5, 1257.8)))
