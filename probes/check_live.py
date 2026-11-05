# -*- coding: utf-8 -*-
import numpy as np, os, sys
fp = r'D:\pyfai-project\runs\refine_v2\results.csv'
if not os.path.isfile(fp):
    print('CSV 还没生成'); sys.exit()
rows = [l.strip().split(',') for l in open(fp, encoding='utf-8').readlines()[1:] if l.strip()]
print(f'已完成 {len(rows)} 张')
a = np.array([[float(x) for x in r[2:]] for r in rows])
dist, eb, ea, secs = a[:, 0], a[:, 1:7], a[:, 7:13], a[:, 13]
names = ('dist', 'poni1', 'poni2', 'rot1', 'rot2', 'rot3')
print(f"{'参数':6s}{'NN粗解':>10s}{'精修后':>10s}")
for i, nm in enumerate(names):
    print(f"{nm:6s}{eb[:, i].mean():10.3f}{ea[:, i].mean():10.3f}")
dv = ea[:, 0] > 2 * np.maximum(eb[:, 0], 1)
print(f'发散 {dv.sum()} 例 / {len(rows)}')
print(f'距离分布: {dist.min():.2f}-{dist.max():.2f}m, 近距(<0.39)比例 {(dist < 0.39).mean() * 100:.0f}%')
m = dist < 0.39
print(f'近距: 粗 {eb[m, 0].mean():.2f} -> 精 {ea[m, 0].mean():.2f} mm (N={m.sum()})')
if (~m).any():
    print(f'中远: 粗 {eb[~m, 0].mean():.2f} -> 精 {ea[~m, 0].mean():.2f} mm (N={(~m).sum()})')
print('最近5张 dist 误差(mm):', ' '.join(f'{b:.1f}->{aa:.1f}' for b, aa in zip(eb[-5:, 0], ea[-5:, 0])))
bad = np.argsort(ea[:, 0] - eb[:, 0])[-5:][::-1]
print('恶化最狠5张 (k, 距离m, 粗->精):')
for i in bad:
    print(f'  k={int(a.shape[0] and rows[i][0])} {rows[i][1]:7s} {dist[i]:.2f}m  {eb[i,0]:.1f} -> {ea[i,0]:.1f}mm')
