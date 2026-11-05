# -*- coding: utf-8 -*-
"""打印基准 JSON 的精度摘要。"""
import json
import os

ROOT = r"D:\pyfai-project"
d = json.load(open(os.path.join(ROOT, "docs", "zoo_benchmark_v3.json"),
                   encoding="utf-8"))
hdr = ("tag", "dD(mm)", "dc_ref", "GT_dc_ref", "lam")
print("{:<14}{:>9}{:>9}{:>10}{:>9}".format(*hdr))
for r in d:
    def fmt(v, p=2):
        return ("%.*f" % (p, v)) if isinstance(v, (int, float)) else "—"
    print("{:<14}{:>9}{:>9}{:>10}{:>9}".format(
        str(r.get("tag")), fmt(r.get("dD")), fmt(r.get("dcenter_ref_px")),
        fmt(r.get("gt_dcenter_ref_px"), 1), fmt(r.get("lam_fit"), 4)))
