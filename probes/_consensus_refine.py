# -*- coding: utf-8 -*-
"""共识硬化精修（10-29 遗留#1）：远距假盆地终局方案。

机理回顾：远距似然谷浅、手工 GT poni 与似然最优 poni 差 ~10px，
单起点/单家族起点落点看运气。方案：
  A 相：dist 网格（偏正，抵粗解低估偏置）× NN poni
  B 相：A 相 top-3 中位 dist × poni 四向抖动 ±12px（覆盖 GT-似然系统差）
  共识：全部终点按 dist 容差 20mm 聚类，类内 min-loss 排序，报告
        类大小/中位/IQR——大且紧的类才可信。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
import fabio
import pyFAI

from real_refine_v5 import to_nn_input_phys
from real_refine_v6 import (REAL, ROOT, load_px_nn, match_gain, nn_coarse,
                            to_nn_input_wide)
from torchfai.refine import DifferentiableRefiner

DIST_GRID = (-32, -16, 0, 16, 32, 48, 64)          # mm，偏正
PONI_JIT = [(0.0, 0.0), (12, 0), (-12, 0), (0, 12), (0, -12)]   # 6M 原生 px
TOL_MM = 20.0

m, dev, consts = load_px_nn(os.path.join(ROOT, "runs/px_v2/best.pt"))


def consensus(finals):
    """finals: list[(loss, dist_m, params)] → (best_cluster 代表解, 诊断)。"""
    finals = sorted(finals, key=lambda f: f[0])
    clusters = []
    for f in finals:                                  # 贪心聚类
        for c in clusters:
            if abs(f[1] - c["med"]) <= TOL_MM / 1000:
                c["members"].append(f)
                c["med"] = float(np.median([x[1] for x in c["members"]]))
                break
        else:
            clusters.append(dict(med=f[1], members=[f]))
    for c in clusters:
        c["minloss"] = min(x[0] for x in c["members"])
        c["iqr"] = (np.percentile([x[1] for x in c["members"]], 75) -
                    np.percentile([x[1] for x in c["members"]], 25)) * 1000
    clusters.sort(key=lambda c: c["minloss"])
    best = clusters[0]
    rep = min(best["members"], key=lambda x: x[0])    # 类内 min-loss 解
    diag = (len(clusters), [(len(c["members"]), round(c["minloss"], 1),
                             round((c["med"] - best["med"]) * 1000, 1),
                             round(c["iqr"], 1)) for c in clusters[:4]])
    return rep, diag


def run_ceria():
    ref6 = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_px1"),
                                 detector="Pilatus6M", wavelength_m=9.72386e-11,
                                 dist_range=[0.1, 1.0])
    px = 172e-6
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        wide, k, extra = to_nn_input_wide(img)
        p_nn, _ = nn_coarse(m, dev, consts, wide, 0.972386, k)
        pon = np.array([p_nn[1], p_nn[2] + extra])    # 6M 原生系 poni

        finals = []
        for dm in DIST_GRID:                          # A 相
            st = np.array([np.clip(p_nn[0] + dm * 1e-3, 0.1, 1.0),
                           pon[0], pon[1], 0, 0, 0])
            r = ref6.refine(img, st, iters=400, sigma_norm=None,
                            sigma_range=(0.0015, 0.02))
            finals.append((r["loss"], r["params"][0], r["params"]))
        top3 = sorted(finals)[:3]
        d1 = float(np.median([f[1] for f in top3]))
        for jy, jx in PONI_JIT[1:]:                   # B 相（四向抖动）
            st = np.array([d1, pon[0] + jy * px, pon[1] + jx * px, 0, 0, 0])
            r = ref6.refine(img, st, iters=400, sigma_norm=None,
                            sigma_range=(0.0015, 0.02))
            finals.append((r["loss"], r["params"][0], r["params"]))

        rep, diag = consensus(finals)
        ed = (rep[1] - gt[0]) * 1000
        print(f"[ceriaW {d}mm] 共识 distΔ={ed:+7.2f}mm  "
              f"poniΔ=({(rep[2][1]-gt[1])/px:+5.1f},"
              f"{(rep[2][2]-gt[2])/px:+5.1f})px  loss={rep[0]:7.2f}  "
              f"聚类(数,minloss,偏距,IQR)={diag}", flush=True)


def run_agbh():
    ref2 = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_px1"),
                                 detector="Pilatus_1M", wavelength_m=1.000e-10,
                                 dist_range=[0.1, 3.0])
    px = 172e-6
    img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
    p = pyFAI.load(os.path.join(REAL, "Pilatus1M.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    p_nn, _ = nn_coarse(m, dev, consts,
                        match_gain(to_nn_input_phys(img)) * 0.5, 1.0, 1.0)
    finals = []
    for dm in DIST_GRID:
        st = np.array([np.clip(p_nn[0] + dm * 1e-3, 0.1, 3.0),
                       p_nn[1], p_nn[2], 0, 0, 0])
        r = ref2.refine(img, st, iters=400, sigma_norm=None,
                        sigma_range=(0.0015, 0.02))
        finals.append((r["loss"], r["params"][0], r["params"]))
    top3 = sorted(finals)[:3]
    d1 = float(np.median([f[1] for f in top3]))
    for jy, jx in PONI_JIT[1:]:
        st = np.array([d1, p_nn[1] + jy * px, p_nn[2] + jx * px, 0, 0, 0])
        r = ref2.refine(img, st, iters=400, sigma_norm=None,
                        sigma_range=(0.0015, 0.02))
        finals.append((r["loss"], r["params"][0], r["params"]))
    rep, diag = consensus(finals)
    ed = (rep[1] - gt[0]) * 1000
    print(f"[AgBh 1M] 共识 distΔ={ed:+7.2f}mm  "
          f"poniΔ=({(rep[2][1]-gt[1])/px:+5.1f},{(rep[2][2]-gt[2])/px:+5.1f})px"
          f"  loss={rep[0]:9.2f}  聚类={diag}", flush=True)


if __name__ == "__main__":
    run_ceria()
    run_agbh()
