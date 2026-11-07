# -*- coding: utf-8 -*-
"""_b7_ab.py:B7 生产 A/B 验证(2026-11-07)——自动缝 mask 对解质量的影响。

帧:Pilatus2M CdTe ID15 CeO2(GT 800mm, λ=12.398/72.1keV=0.17195Å,
px=172µm)+ ceria_200(Frelon 阴性回归,GT 200.19mm)。
口径:auto on/off 两跑,比 |ΔD|、ps、m、rms;并断言 mask 像素数>0(P2M)。
"""
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
import os
os.chdir(ROOT)

import solver_core as sc
from gui_solver import load_any          # 生产同款载入(zoo 预处理在内)

LAM = 12.398 / 72.1


def run(tag, path, gt_mm, px_um=172.0, lam=LAM):
    img = load_any(path)
    out = {}
    for label, flag in (("auto-on", True), ("auto-off", False)):
        sc._AUTO_GAP = flag
        sc.clear_masks()
        r = sc.solve_image(img, px_um * 1e-6, lam=lam)
        if r is None:
            out[label] = None
            continue
        D = r["sol"][2] * px_um * 1e-3 * r["s"] if r["s"] else \
            r["sol"][2] * px_um * 1e-3
        # sol 是画布坐标 → 原生:D_px_native = sol[2]*s
        D_mm = r["sol"][2] * r["s"] * px_um * 1e-3
        out[label] = dict(ph=r["phase"], D=D_mm, ps=r["ps"], m=r["m"],
                          rms=r["rms"], gap=r.get("gap_masked", 0))
    sc._AUTO_GAP = True
    print(f"\n== {tag} GT={gt_mm}mm ==")
    for label in ("auto-on", "auto-off"):
        d = out[label]
        if d is None:
            print(f"  {label:9} 解不出")
            continue
        print(f"  {label:9} ph={d['ph']:6} D={d['D']:8.2f} "
              f"Δ={abs(d['D']-gt_mm):6.2f}  ps={d['ps']:.3f} m={d['m']:2} "
              f"rms={d['rms']:.2f} gap_masked={d['gap']}")
    return out


r1 = run("Pilatus2M CdTe CeO2 800mm",
         ROOT + r"\data\real\Pilatus2MCdTe_ID15_CeO2_72100eV_800mm_0000.cbf",
         800.0, lam=LAM)
r2 = run("ceria_200(Frelon 阴性)",
         ROOT + r"\data\real\MX_ceria\ceria_200_1_0001.cbf",
         200.19, lam=0.972386)

ok = True
# P2M:两跑都应解出;auto-on 不劣于 off(gap 琏不再污染径向谱)
for lbl in ("auto-on", "auto-off"):
    if r1[lbl] is None:
        print(f"!! P2M {lbl} 解不出")
        ok = False
if r1["auto-on"]:
    assert r1["auto-on"]["gap"] > 0, "P2M 应检出缝"
    d_on = abs(r1["auto-on"]["D"] - 800.0)
    d_off = abs(r1["auto-off"]["D"] - 800.0) if r1["auto-off"] else 9e9
    print(f"\nP2M: |ΔD| on={d_on:.2f} off={d_off:.2f} "
          f"({'on 更优/平' if d_on <= d_off + 1.0 else 'on 劣化!'})")
# ceria:两跑都解出且 D 一致(mask 只动了边界死区)
if r2["auto-on"] and r2["auto-off"]:
    dd = abs(r2["auto-on"]["D"] - r2["auto-off"]["D"])
    print(f"ceria: on/off D 差 {dd:.2f}mm "
          f"(gap={r2['auto-on']['gap']}px 死区)")
    assert dd < 2.0, "ceria 上 auto mask 改变了解(不应发生)"
print("\nB7 生产接线 A/B:", "通过" if ok else "有异常,看上表")
