# -*- coding: utf-8 -*-
"""duel v2 → v3 增量：ceria 行 T 臂补原生精化（rg2，zoo T7 同机器）。

v2 里 ceria 行走 q_list 锁表模式，solve_cascade 不带 2D 尾 → T7 门
（sol[10]）不触发 → T 臂中心停在画布级（Δc 8–27px），而 zoo 标准行
（P2M/E9M）带 ⁿ 尾、生产管线（phase_fit）另有 fit_tilt 下游。为让
T 臂代表"发货管线"，锁表行无条件跑 refine_geometry_2d（q_list 的 8
环表），P1 用新 T 重初始化。zoo 行保持 v2 不动（口径=zoo 标准行）。
输出：pyfai_duel_v3.md/.json（v2 保留原档）。
"""
import os
import sys
import time
import json

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import _pyfai_duel as D                      # noqa: E402（复用全部部件）
from solver_core import solve_image          # noqa: E402
from solver_core import refine_geometry_2d, concentric_center  # noqa: E402
from torchfai.phases import phase_qs         # noqa: E402

OUT = os.path.join(ROOT, "probes", "results")
CERIA = os.path.join(ROOT, "data", "real", "MX_ceria")


def ours_run2(img, px_native, lam_A, q_list):
    """T 臂（锁表+无条件原生精化=生产代表）。"""
    t0 = time.perf_counter()
    res = solve_image(img, px_native, lam=lam_A, q_list=q_list)
    if res is None:
        return None
    s, px_m = res["s"], res["px_m"]
    sol = list(res["sol"])
    lam_used = res["lam"]
    o = None
    try:
        o = refine_geometry_2d(img, sol[0] * s, sol[1] * s, sol[2] * s,
                               np.asarray(q_list, float), lam_used, n_iter=3)
    except Exception:
        o = None
    tier_n = False
    if o is not None:
        sol[0], sol[1], sol[2] = o[0] / s, o[1] / s, o[2] / s
        tier_n = True
    cx, cy, D_px, m = sol[0], sol[1], sol[2], sol[3]
    return dict(cx=float(cx * s), cy=float(cy * s),
                D_mm=float(D_px * px_m * 1000.0), m=int(m),
                lam=float(lam_used) if lam_used else None,
                ms=(time.perf_counter() - t0) * 1000,
                fb=bool(res["used_fallback"]), fbr=str(res["fb_reason"]),
                tier=(f"s={s}" + ("ⁿ" if tier_n else "")))


def main():
    with open(os.path.join(OUT, "pyfai_duel_v2.json"), encoding="utf-8") as f:
        rows = json.load(f)
    Q8 = phase_qs("CeO2")[:8]
    for r in rows:
        if not str(r.get("tag", "")).startswith("ceria"):
            continue
        d = int(r["tag"][5:])
        f = f"ceria_{d}_1_0001"
        img = D.load_img(os.path.join(CERIA, f + ".cbf"))
        ai, gx, gy, gD = D.gt_of(os.path.join(CERIA, f + ".poni"))
        det = ai.detector
        pxn = float(det.pixel1)
        lam_A = float(ai.wavelength) * 1e10

        t = ours_run2(img, pxn, lam_A, Q8)
        t["dc_gt"] = float(np.hypot(t["cx"] - gx, t["cy"] - gy))
        t["dc_ref"] = D.cref_dc(img, t["cx"], t["cy"], t["D_mm"],
                                "CeO2", t["lam"], pxn)
        r["T"] = t
        p1 = D.pyfai_run(img, det, "CeO2", lam_A, t["D_mm"] / 1000.0,
                         t["cx"], t["cy"])
        if "cx" in p1:
            p1["dc_gt"] = float(np.hypot(p1["cx"] - gx, p1["cy"] - gy))
            p1["dc_ref"] = D.cref_dc(img, p1["cx"], p1["cy"], p1["D_mm"],
                                     "CeO2", p1.get("lam") or lam_A, pxn)
        r["P1"] = p1
        print(f"[{r['tag']}] T={t['D_mm']-gD:+.2f}/{t['dc_gt']:.1f} "
              f"P1={p1.get('D_mm', float('nan'))-gD:+.2f}/"
              f"{p1.get('dc_gt', float('nan')):.1f}")

    # ---- 重建表格（与 v2 同款列）----
    md = ["# pyFAI 对照（duel v3）——同图同 λ 同标样，标准流程对决",
          "",
          "v3 = v2 + ceria 行 T 臂原生精化（rg2/同 zoo T7 机器；锁表模式"
          "画布解不带 2D 尾，v2 少了这一步属探针口径缺口）。",
          "T=solve_image 统一入口（ceria 行锁表 8 环+原生精化）；"
          "P0=pyFAI RingExtraction+refine3 零起点（D 网网格取其 chi2 最优，"
          "中心=几何中心）；P1=同 P0 但以 T 解为初值。"
          "Frelon2k/halfccd GT poni 无波长 → λ 联合模式。",
          "ΔD 单位 mm；ΔλD=λ̂·D̂ 相对 GT 偏差%（λ–D 脊上≈0=几何正确）；"
          "Δc 原生 px（ᴳᵀ=GT 口径，ᶜᵒⁿ=共心真值 T9）。*P0 耗时=三网格总和。",
          "",
          "| 项 | 标样 | 臂 | ΔD(mm) | ΔλD(%) | Δcᴳᵀ | Δcᶜᵒⁿ | λ̂(Å) | 点/环 | 耗时 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            md.append(f"| {r['tag']} | ERROR | | | | | | | | |")
            continue
        gt_D, lam_gt = r["gt_D"], r.get("lam_gt")
        for arm, label in (("T", "T"), ("P0", "P0"), ("P1", "P1")):
            a = r.get(arm)
            c = D.cell(a, gt_D, lam_gt)
            tier = (a or {}).get("tier", "")
            if arm == "T":
                md.append(f"| {r['tag']} | {r['cal']} | {label} {tier} | "
                          + " | ".join(c) + f" | {a.get('ms',0)/1000:.0f}s |")
            else:
                md.append(f"|  |  | {label} | " + " | ".join(c)
                          + f" | {a.get('ms',0)/1000:.0f}s{'*' if arm=='P0' else ''} |")
    with open(os.path.join(OUT, "pyfai_duel_v3.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "pyfai_duel_v3.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
    print("saved -> probes/results/pyfai_duel_v3.md/.json")


if __name__ == "__main__":
    main()
