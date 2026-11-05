# -*- coding: utf-8 -*-
"""八件套消融（合成压力集）：每个组件是否可辨识，取决于数据是否触发它。

压力集：D 100–600mm、倾角 0–3°、曝光 300–10000、束心 0.35–0.65（离轴）——12 样本。
臂：A 完整 / B 关2D / C 关自动定心 / D 关级联 / E 喂GT相 / F λ未知 / G 加倾角层。
"""
import os
import sys
import time

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from torchfai.simulator import get_detector, synth_sample
from torchfai.config import SimConfig
from torchfai.phases import PHASES
from solver_core import to_canvas, solve_auto, solve_auto_lambda

OUT = os.path.join(ROOT, "probes", "results")
LAM = 0.5
PH = "CeO2"
DET = ("Pilatus_1M", 2)
N = 40


def cfg():
    return SimConfig(detector_name=DET[0], binning=DET[1], wavelength=LAM * 1e-10,
                     dist_range=(0.10, 0.60), poni_frac_range=(0.35, 0.65),
                     rot1_range=(-np.radians(3), np.radians(3)),
                     rot2_range=(-np.radians(3), np.radians(3)),
                     rot3_range=(0.0, 0.0),
                     peak_sigma_range=(0.004, 0.010), amp_jitter=(0.7, 1.4),
                     dw_b_range=(0.0, 0.01), bg0_range=(0.0, 0.01),
                     bg1_range=(0.0, 0.005), hump_prob=0.3,
                     peak_counts_range=(300, 10000), peak_counts_loguniform=True,
                     hot_rate_range=(0.0, 1e-4), dead_rate_range=(0.0, 0.0),
                     beamstop_prob=0.5, min_visible_peaks=3)


def main():
    db = get_detector(*DET)
    idx = list(PHASES).index(PH)
    arms = ["A", "B", "C", "D", "E", "F", "G"]
    rows = {a: [] for a in arms}
    t0 = time.time()
    for k in range(N):
        rng = np.random.default_rng(500 + k)
        s = synth_sample(rng, cfg(), db, phase_idx=idx)
        img = s["img"].astype(np.float32)
        gt = s["geom"]
        canvas, sc, _, _ = to_canvas(img)
        pxm = db.pixel * sc
        for a in arms:
            try:
                if a == "F":
                    r = solve_auto_lambda(canvas, pxm, auto_center=True)
                    if r is None:
                        rows[a].append(None); continue
                    ph, sol, lam = r
                else:
                    kw = dict(cascade=True, auto_center=True, refine_2d=True)
                    if a == "B":
                        kw["refine_2d"] = False
                    elif a == "C":
                        kw["auto_center"] = False
                    elif a == "D":
                        kw["cascade"] = False
                    elif a == "E":
                        kw["phases"] = [PH]
                    r = solve_auto(canvas, LAM, pxm, **kw)
                    if r is None:
                        rows[a].append(None); continue
                    ph, sol = r
                dD = sol[2] * pxm * 1000 - gt["dist"] * 1000
                gcx = gt["poni2"] / db.pixel - 0.5
                gcy = gt["poni1"] / db.pixel - 0.5
                dc = float(np.hypot(sol[0] * sc - gcx, sol[1] * sc - gcy))
                rows[a].append(dict(dD=dD, dc=dc, ok=(ph == PH)))
            except Exception:
                rows[a].append(None)
        print(k, {a: (None if rows[a][-1] is None else
                      f"{rows[a][-1]['dD']:+.1f}/{rows[a][-1]['dc']:.0f}px/{'OK' if rows[a][-1]['ok'] else 'x'}")
                for a in arms}, flush=True)

    md = ["# 八件套消融（合成压力集，12 样本）", "",
          "条件：D 0.1–0.6m、倾角 ±3°、曝光 300–10000、束心 0.35–0.65、束流挡盘 50%、驼峰 30%", "",
          "| 臂 | 说明 | 解出 | 中位\\|ΔD\\| | 最差\\|ΔD\\| | 中位Δc | 相正确 |", "|---|---|---|---|---|---|---|"]
    names = dict(A="完整", B="关 2D 精化", C="关自动定心", D="关级联",
                 E="喂 GT 相（关仲裁）", F="λ 未知", G="加倾角层")
    for a in arms:
        rs = [r for r in rows[a] if r]
        if not rs:
            md.append(f"| {a} | {names[a]} | 0/{N} | — | — | — | — |"); continue
        dD = np.abs([r["dD"] for r in rs]); dc = np.array([r["dc"] for r in rs])
        q1,q3=np.percentile(dD,[25,75])
        md.append(f"| {a} | {names[a]} | {len(rs)}/{N} | {np.median(dD):.2f} | {dD.max():.2f} | "
                  f"{np.median(dc):.1f} | {np.mean([r['ok'] for r in rs])*100:.0f}% |")
    md += ["", f"总耗时 {time.time()-t0:.0f}s"]
    ts = time.strftime("%Y%m%d_%H%M")
    p = os.path.join(OUT, f"ablation8_syn_{ts}.md")
    open(p, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md)); print("saved", p)


if __name__ == "__main__":
    main()

