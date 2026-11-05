# -*- coding: utf-8 -*-
"""精修后全参数精度统计（refine_v2 全量 3900），换算像素单位。"""
import sys
import json

sys.path.insert(0, r"D:\pyfai-project")
import numpy as np
from torchfai.simulator import get_detector

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cfg = json.load(open(r"D:\pyfai-project\data\train_v2\meta.json",
                     encoding="utf-8"))["config"]
db = get_detector(cfg["detector_name"], cfg["binning"])
print(f"detector={cfg['detector_name']} binning={cfg['binning']} "
      f"pixel={db.pixel*1e6:.1f}um extent={db.extent}")
print()

rows = [l.strip().split(",")
        for l in open(r"D:\pyfai-project\runs\refine_v2\results.csv",
                      encoding="utf-8").readlines()[1:]]
a = np.array([[float(x) for x in r[2:]] for r in rows])
dist, eb, ea = a[:, 0], a[:, 1:7], a[:, 7:13]
px = db.pixel * 1e3  # mm per pixel
names = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
print(f"{'param':6s} {'coarseMAE':>10s} {'refMAE':>9s} {'refMED':>9s} {'refP95':>9s}   (mm/mrad)")
for j, nm in enumerate(names):
    print(f"{nm:6s} {eb[:, j].mean():10.3f} {ea[:, j].mean():9.3f} "
          f"{np.median(ea[:, j]):9.3f} {np.percentile(ea[:, j], 95):9.3f}")
print()
ok = ea[:, 0] < 5
for lab, m in (("全样本", np.ones(len(a), bool)), ("非卡住(精修<5mm)", ok)):
    print(f"{lab} N={m.sum()}: dist 中位 {np.median(ea[m,0]):.3f}mm "
          f"({np.median(ea[m,0])/px:.2f}px) | poni1 {np.median(ea[m,1]):.3f}mm "
          f"({np.median(ea[m,1])/px:.2f}px) | poni2 {np.median(ea[m,2]):.3f}mm "
          f"({np.median(ea[m,2])/px:.2f}px)")
    print(f"      dist 相对误差中位 {np.median(ea[m,0]*1e-3/dist[m])*100:.4f}%  "
          f"poni 相对半视场 {np.median(ea[m,1]/(db.extent[0]*1e3/2))*100:.3f}%")
