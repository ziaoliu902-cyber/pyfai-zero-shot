# -*- coding: utf-8 -*-
"""从 refine_v2 结果里找免费精度：符号偏置、精修位移量、参数间耦合。"""
import numpy as np
import sys

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = [l.strip().split(",")
        for l in open(r"D:\pyfai-project\runs\refine_v2\results.csv",
                      encoding="utf-8").readlines()[1:]]
a = np.array([[float(x) for x in r[2:]] for r in rows])
dist, eb, ea = a[:, 0], a[:, 1:7], a[:, 7:13]
ok = ea[:, 0] < 5          # 非卡住
names = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]

print("=== 1) 符号偏置（非卡住 N=%d，MAE%%列=|bias|/MAE）===" % ok.sum())
for j, nm in enumerate(names[:5]):
    bias = ea[ok, j].mean()
    mae = np.abs(ea[ok, j]).mean()
    print(f"{nm:5s} bias={bias:+8.3f}  MAE={mae:7.3f}  "
          f"|bias|/MAE={abs(bias)/mae*100:5.1f}%   (mm/mrad)")

print()
print("=== 2) 精修到底动了多少参数（非卡住）===")
print(f"{'param':6s} {'粗解中位|e|':>10s} {'精修中位|e|':>10s} "
      f"{'中位|位移|':>10s} {'位移/粗误差':>10s}")
for j, nm in enumerate(names[:5]):
    move = np.abs(ea[ok, j] - eb[ok, j])
    print(f"{nm:6s} {np.median(np.abs(eb[ok, j])):10.3f} "
          f"{np.median(np.abs(ea[ok, j])):10.3f} {np.median(move):10.3f} "
          f"{np.median(move)/np.median(np.abs(eb[ok, j]))*100:9.0f}%")

print()
print("=== 3) 精修后误差相关（非卡住，看参数耦合）===")
C = np.corrcoef(ea[ok, :4].T)
lbl = ["dist ", "poni1", "poni2", "rot1 "]
print("        " + "  ".join(lbl))
for i, l1 in enumerate(lbl):
    print(l1 + "  " + "  ".join(f"{C[i,j]:+.2f}" for j in range(4)))

print()
print("=== 4) 精修后 vs 粗解误差相关（精修是独立信息还是跟随粗解？）===")
for j, nm in enumerate(names[:5]):
    r = np.corrcoef(eb[ok, j], ea[ok, j])[0, 1]
    print(f"{nm:5s} corr(粗误差, 精修误差) = {r:+.3f}")

print()
print("=== 5) 多环样本（代理：精修误差<1mm 的）占比 ===")
for th in (1, 0.5, 0.2):
    print(f"  精修 dist < {th}mm: {(np.abs(ea[:,0])<th).mean()*100:.1f}%")
