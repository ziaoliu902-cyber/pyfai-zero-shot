# -*- coding: utf-8 -*-
import numpy as np
fp = r'D:\pyfai-project\runs\refine_v2\results.csv'
rows = [l.strip().split(',') for l in open(fp, encoding='utf-8').readlines()[1:] if l.strip()]
a = np.array([[float(x) for x in r[2:]] for r in rows])
ph = np.array([r[1] for r in rows])
k = np.array([int(r[0]) for r in rows])
dist, eb, ea = a[:, 0], a[:, 1:7], a[:, 7:13]

m = dist >= 0.39
print(f'== 中远距 {m.sum()} 张 ==')
print(f'dist: 均值 粗{eb[m,0].mean():.1f}->精{ea[m,0].mean():.1f} | 中位数 粗{np.median(eb[m,0]):.1f}->精{np.median(ea[m,0]):.1f}mm')
for lo, hi, tag in ((0.39,0.77,'中距'), (0.77,9,'远距')):
    mm = (dist>=lo)&(dist<hi)
    if mm.sum():
        print(f'{tag} N={mm.sum():3d}: 中位数 粗{np.median(eb[mm,0]):6.1f} -> 精{np.median(ea[mm,0]):6.2f}mm | >10mm残余 {(ea[mm,0]>10).sum()}例')
print()
print('残余>10mm 的样本（按相分组）:')
bad = (ea[:,0]>10)
for p in ('LaB6','CeO2','AgBh'):
    mm = bad & (ph==p)
    if mm.sum():
        print(f'  {p:5s} {mm.sum():3d}例: 距离 {dist[mm].min():.2f}-{dist[mm].max():.2f}m, '
              f'粗解误差中位 {np.median(eb[mm,0]):.0f}mm')
print()
print('结论检查: 这些钉子户粗解本来就差吗？')
print(f'  残余>10mm例中, 粗解误差中位数 = {np.median(eb[bad,0]):.0f}mm (若本身就>10mm说明是NN粗解差, 不是精修的锅)')
print(f'  精修后仍变差的比例: {(ea[bad,0]>eb[bad,0]).mean()*100:.0f}%')
