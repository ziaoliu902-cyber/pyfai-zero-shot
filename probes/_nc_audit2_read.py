# -*- coding: utf-8 -*-
"""_nc_audit2_read.py — 审计 v0.1 结果分层统计。"""
import json

import numpy as np

j = json.load(open(r"D:\pyfai-project\docs\nc_audit_v0_1.json",
                   encoding="utf-8"))
ok = [r for r in j if "dc_v3" in r]
al = [r for r in ok if r["tag"].startswith("alumina")]
zoo = [r for r in ok if not r["tag"].startswith("alumina")]

print(f"== zoo/ceria 族（{len(zoo)} 对）==")
for r in zoo:
    g = r["gt_dc"] if r["gt_dc"] is not None else "—"
    print(f"  {r['tag']:14} {r['tier']} {r['arm']} {r['ph']:7} "
          f"Δc(V3)={r['dc_v3']:7.2f}  GTΔc(ref)={g:>7}  仲裁={r['verdict']}")

print(f"\n== alumina 族（{len(al)} 对）==")
vd = {}
for r in al:
    vd[r["verdict"]] = vd.get(r["verdict"], 0) + 1
print("  仲裁分布:", vd)
print("  Δc(V3) 中位 %.2f px | ≤1px %d 帧 | ≤2.4px %d 帧 | >10px %d 帧" % (
    np.median([r["dc_v3"] for r in al]),
    sum(1 for r in al if r["dc_v3"] <= 1),
    sum(1 for r in al if r["dc_v3"] <= 2.4),
    sum(1 for r in al if r["dc_v3"] > 10)))
print("  相鉴: Al2O3 %d / 其他 %d" % (
    sum(1 for r in al if r["ph"] == "Al2O3"),
    sum(1 for r in al if r["ph"] != "Al2O3")))
good = [r for r in al if r["ph"] == "Al2O3"]
if good:
    print("  仅 Al2O3✔帧: Δc(V3) 中位 %.2f px | ≤2.4px %d/%d" % (
        np.median([r["dc_v3"] for r in good]),
        sum(1 for r in good if r["dc_v3"] <= 2.4), len(good)))
    vg = {}
    for r in good:
        vg[r["verdict"]] = vg.get(r["verdict"], 0) + 1
    print("  仅 Al2O3✔帧 仲裁分布:", vg)
T = [r for r in ok if r["tier"] == "T"]
t = [r for r in ok if r["tier"] == "t"]
print(f"\n== 分层汇总 == T组 {len(T)} / t组 {len(t)}")
for nm, grp in (("T(≥2mrad)", T), ("t(<2mrad)", t)):
    if grp:
        vd = {}
        for r in grp:
            vd[r["verdict"]] = vd.get(r["verdict"], 0) + 1
        print(f"  {nm}: Δc(V3) 中位 {np.median([r['dc_v3'] for r in grp]):.2f}px"
              f"  仲裁 {vd}")
