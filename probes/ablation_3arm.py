# -*- coding: utf-8 -*-
"""ablation_3arm.py：消融三臂（B 路线 Phase 4b，论文核心 Table 素材）。

同一画布、同一峰提取、同一 LSQ——只换初始化，隔离 NN 粗解的贡献：
- A 纯NN    : v4s1 直接回归 6 参数，无任何经典精化
- B 纯经典  : 无 NN——中心=画布几何中心，D 全域投票(d_range 0.05~3.0)，
              盆域门关闭（无先验可门）
- C 混合    : 生产路径——NN 初始化(中心+D) + 盆域门(±80mm) + 经典精化

判据：|ΔD| vs GT（预注册口径 mean<10 / worst<15）；成功=|ΔD|<15mm。
用法: python ablation_3arm.py
"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI
import torch

from real_refine_v6 import load_px_nn, to_nn_input_wide
from torchfai.dataset import TorchFAIDataset as _T
from torchfai.dataset import norm_to_param
from solver_core import solve, ring_radii
from torchfai.phases import phase_qs

LAM = 0.972386
PXEFF = 172e-6 / (521.0 / 2527)
Q = phase_qs("CeO2")[:8]
ds = _T(os.path.join(ROOT, "data", "train_v4w3"), "val")
consts = ds.norm_consts
m1, dev, _ = load_px_nn(r"runs\v4s1\best.pt")
m1.eval()

print("== ceria 消融三臂：A=纯NN  B=纯经典(无先验)  C=混合(生产) ==")
print(f"{'帧':>5}{'GT':>8}{'AΔmm':>8}{'BΔmm':>9}{'B环':>4}{'Brms':>6}"
      f"{'CΔmm':>8}{'C环':>4}{'Crms':>6}")
res = {a: [] for a in "ABC"}
tA = tB = tC = 0.0
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    wide, k, _ = to_nn_input_wide(img)
    Hm, Wm = wide.shape
    pxm = PXEFF * 521.0 / Hm
    x = torch.from_numpy(
        np.log1p(wide[None, None].astype(np.float32))).to(dev)

    # ---- A: 纯 NN ----
    t0 = time.perf_counter()
    with torch.no_grad():
        mu, _ = m1(x, torch.tensor([LAM], device=dev),
                   torch.tensor([k], device=dev))
    p6 = norm_to_param(mu.double().cpu().numpy(), consts, k)[0]
    tA += time.perf_counter() - t0
    eA = p6[0] * 1000 - gt
    res["A"].append(abs(eA))

    # ---- B: 纯经典（无 NN：中心=几何中心，全域，无盆域门）----
    t0 = time.perf_counter()
    cxn, cyn = (Wm - 1) / 2.0, (Hm - 1) / 2.0
    pkn = ring_radii(wide, cxn, cyn)[0]
    solB = solve(pkn, cxn, cyn, 1.0 / pxm, Q, LAM, pxm,
                 d_range=(0.05, 3.0), rms_floor=1.8)
    tB += time.perf_counter() - t0
    if solB is None:
        print(f"{d:>5}{gt:8.1f}{eA:+8.1f}{'失败':>9}")
        res["B"].append(999.0)
        eB = np.nan
    else:
        eB = solB[2] * pxm * 1000 - gt
        res["B"].append(abs(eB))

    # ---- C: 混合（生产路径：NN 初始化 + 盆域门）----
    t0 = time.perf_counter()
    cx0 = p6[2] / PXEFF * (Wm / 490.0)
    cy0 = p6[1] / PXEFF * (Hm / 521.0)
    D_px0 = p6[0] / PXEFF * (Hm / 521.0)
    pkc = ring_radii(wide, cx0, cy0)[0]
    solC = solve(pkc, cx0, cy0, D_px0, Q, LAM, pxm,
                 D_ref_m=p6[0], d_range=(0.35, 2.8), rms_floor=1.8)
    tC += time.perf_counter() - t0
    if solC is None:
        print(f"{d:>5}{gt:8.1f}{eA:+8.1f}{eB:+9.1f}{'C失败':>8}")
        res["C"].append(999.0)
        continue
    eC = solC[2] * pxm * 1000 - gt
    res["C"].append(abs(eC))
    bmsg = (f"{eB:+9.1f}{solB[3]:>4}{solB[4]:6.2f}" if solB else
            f"{'失败':>9}")
    print(f"{d:>5}{gt:8.1f}{eA:+8.1f}{bmsg}{eC:+8.1f}{solC[3]:>4}"
          f"{solC[4]:6.2f}")

print(f"\n{'臂':>10}{'平均|Δ|':>10}{'最差':>8}{'成功(<15mm)':>12}"
      f"{'耗时/帧':>10}")
for a, t_ in zip("ABC", (tA, tB, tC)):
    v = np.array(res[a])
    ok = int((v < 15).sum())
    print(f"{a:>10}{v.mean():10.1f}{v.max():8.1f}{ok:>10}/7"
          f"{t_/7*1000:9.0f}ms")
print("\n判据(均值<10 最差<15): " + " ".join(
    f"{a}={'✓' if v.mean() < 10 and v.max() < 15 else '✗'}"
    for a, v in ((a, np.array(res[a])) for a in "ABC")))
