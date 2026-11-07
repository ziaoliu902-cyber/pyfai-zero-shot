# -*- coding: utf-8 -*-
"""_b10_fc_phase.py:B10 Fairchild 三级流水·里程碑1——多相线表 2θ 比例
判决(相指认),2026-11-07。

背景:Fairchild(4096²,rot1=-29°束心画布外)生产 SOLVE FAIL;判决实验C
定调救援=相仲裁(多相线表)→RANSAC 指派→倾斜感知拟合。里程碑1:
用 GT poni 积分出的 1D 峰位(相但未知),做跨相库线表比例匹配,看
top-1 是否指认正确(zoo 答案卡:Al2O3,4 环)。

口径:一对一贪心指派(相对容差 1.2%);score=匹配峰数−0.15×赝配
(库线无观测);并列看容差内残差 rms。
"""
import os
import sys

import numpy as np
import fabio
import pyFAI
from scipy.signal import find_peaks

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

from torchfai.phases import PHASES, phase_qs

Z = os.path.join(ROOT, "data", "real", "silx_zoo")
img = fabio.open(Z + r"\Fairchild.edf").data.astype(np.float64)
ai = pyFAI.load(Z + r"\Fairchild.poni")
lam = ai.wavelength
print(f"Fairchild {img.shape} D={ai.dist*1e3:.1f}mm "
      f"λ={lam*1e10:.4f}Å px={ai.detector.pixel1*1e6:.1f}µm")

q, I = ai.integrate1d(img, 16384, unit="q_A^-1",
                      method=("no", "csr", "cpu"))
I = np.nan_to_num(I)
prom = 0.004 * float(np.percentile(I, 99))
pk, props = find_peaks(I, prominence=prom, distance=3)
order = np.argsort(props["prominences"])[::-1][:30]
q_obs = np.sort(q[pk[order]])
q_obs = q_obs[q_obs > np.percentile(q, 2)]
# 剔除近邻重复(合并 <0.8% 的峰)
dedup = []
for x in q_obs:
    if not dedup or x / dedup[-1] > 1.008:
        dedup.append(x)
q_obs = np.array(dedup)
print(f"1D 峰 {len(q_obs)} 个: " +
      " ".join(f"{x:.4f}" for x in q_obs))


def match_phase(name, qo, tol=0.012):
    """一对一贪心指派。返回 (n_match, rms_rel, n_missing)。"""
    lib = np.asarray(phase_qs(name), float)
    lib = lib[(lib > qo.min() * 0.98) & (lib < qo.max() * 1.02)]
    pairs = []
    for i, qi in enumerate(qo):
        if len(lib) == 0:
            break
        j = int(np.argmin(np.abs(lib - qi)))
        r = abs(lib[j] - qi) / qi
        if r < tol:
            pairs.append((r, i, j))
    pairs.sort()
    used_o, used_l, tot_r = set(), set(), 0.0
    for r, i, j in pairs:
        if i in used_o or j in used_l:
            continue
        used_o.add(i); used_l.add(j); tot_r += r * r
    nm = len(used_o)
    rms = float(np.sqrt(tot_r / nm)) if nm else 9.9
    return nm, rms, len(lib) - len(used_l)


print("\n== 多相线表 2θ 比例匹配(λ 无关:q 比值) ==")
rows = []
for name in PHASES:
    nm, rms, miss = match_phase(name, q_obs)
    score = nm - 0.15 * miss
    rows.append((name, nm, rms, miss, score))
    print(f"  {name:8} 匹配 {nm:2}/{len(q_obs)}  残差rms {rms*100:5.2f}%  "
          f"库线未观测 {miss:2}  score={score:5.2f}")
rows.sort(key=lambda r: -r[4])
top = rows[0]
second = rows[1] if len(rows) > 1 else None
print(f"\nTop-1: {top[0]} (score {top[4]:.2f})  "
      f"Top-2: {second[0] if second else '-'} ({second[4] if second else 0:.2f})")
ok = top[0] == "Al2O3"
print(f"\n里程碑1(相指认 top-1=Al2O3): "
      f"{'达成 ✓' if ok else '未达成 ✗(看表,若 top-2=Al2O3 且差距<1 也算弱达成)'}")
