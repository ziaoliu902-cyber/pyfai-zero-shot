# -*- coding: utf-8 -*-
import numpy as np
fp = r'D:\pyfai-project\runs\refine_v2\results.csv'
rows = [l.strip().split(',') for l in open(fp, encoding='utf-8').readlines()[1:] if l.strip()]
a = np.array([[float(x) for x in r[2:]] for r in rows])
dist, eb, ea = a[:, 0], a[:, 1:7], a[:, 7:13]
names = ('dist', 'poni1', 'poni2', 'rot1', 'rot2', 'rot3')
print(f'N={len(rows)}  (跑至第 {rows[-1][0]} 张)')
print(f"{'参数':6s}{'验收线':>8s}{'NN均值':>9s}{'精修均值':>9s}{'精修中位':>9s}")
lines = ('厘米级', '毫米级', '毫米级', 'mrad级', 'mrad级', '—')
for i, nm in enumerate(names):
    print(f"{nm:6s}{lines[i]:>8s}{eb[:,i].mean():9.2f}{ea[:,i].mean():9.2f}"
          f"{np.median(ea[:,i]):9.2f}")
print()
for lo, hi, tag in ((0, 0.39, '近距'), (0.39, 0.77, '中距'), (0.77, 9, '远距')):
    m = (dist >= lo) & (dist < hi)
    if m.sum():
        print(f"{tag} N={m.sum():4d}  dist中位 {np.median(eb[m,0]):6.2f} -> {np.median(ea[m,0]):6.3f}mm"
              f"   poni1中位 {np.median(eb[m,1]):.2f}->{np.median(ea[m,1]):.3f}mm"
              f"   rot1中位 {np.median(eb[m,3]):.2f}->{np.median(ea[m,3]):.2f}mrad")
