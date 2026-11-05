# -*- coding: utf-8 -*-
"""offset_duel.py：离轴束心对决——NN 的主场验证（B 路线 Phase 4c）。

问题（1031 消融引出）：束心贴几何中心时经典路径碾压 NN 初始化
（0.2/0.3mm vs 4.1/8.0mm）——NN 初始化是否在离轴域有不可替代价值？

设计：train_v4w3 val 域（poni 6%~94% 全域采样）按束心离轴量分箱，
同图跑两臂（同 LSQ/同 rms_floor/同相表，只换初始化）：
- B 经典: 几何中心 + 全域投票(d_range 0.05~3.0) + 谐波定心自举
- C 混合: v4s1 NN 初始化 + 盆域门(±80mm) + d_range 0.35~2.8

成功判据：|ΔD|/D < 1% 且配对环 ≥3。n_vis≥4 的样本入选（排除环贫乏
对称失败，聚焦几何难度）。预测：离轴越大 B 崩、C 守——若成立，NN
价值=离轴兜底，classic-first/NN-fallback 架构闭环。
用法: python offset_duel.py [n_per_bin]
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
import torch

from real_refine_v6 import load_px_nn
from solver_core import _r_model, solve, ring_radii
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.phases import PHASES, PHASE_NAMES, phase_qs

N_PER_BIN = int(sys.argv[1]) if len(sys.argv) > 1 else 60
# 离轴分布（val 实测）：中位 0.43×画布，75 分位 1.10，最大 2.3
# （rot±0.07rad × D≤3m 使环心可偏 poni 达 ~600px → 分箱按真环心）
BINS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.45), (0.45, 2.5)]
PX = 344e-6          # Pilatus_1M binned2 两轴同


def beam_center_px(dist, poni1, poni2, rot1, rot2):
    """光轴穿击点（=环心）像素索引。解析解自 tth_of 前向（tth=0 点）：
    v2 = −D·tan(r1)，v1 = tan(r2)·D/cos(r1)。像素中心约定
    phys=(idx+0.5)·px（与 pyFAI/ring_radii 索引一致）。
    倾角大时环心≠poni（rot·D 耦合，可差数百 px）。"""
    v2 = -dist * np.tan(rot1)
    v1 = np.tan(rot2) * dist / np.cos(rot1)
    return (poni2 + v2) / PX - 0.5, (poni1 + v1) / PX - 0.5


# 公式自检：该点 tth 必须为 0（随机几何对拍，硬失败）
from poni_out import tth_of as _tth
_r = np.random.default_rng(0)
_w = 0.0
for _ in range(200):
    _d = 10 ** _r.uniform(-1, np.log10(3.0))
    _p1, _p2 = _r.uniform(0, 0.2, 2)
    _r1, _r2 = _r.uniform(-0.07, 0.07, 2)
    _cx, _cy = beam_center_px(_d, _p1, _p2, _r1, _r2)
    _w = max(_w, abs(float(_tth((_cy + 0.5) * PX, (_cx + 0.5) * PX, _d,
                                _p1, _p2, _r1, _r2))))
assert _w < 1e-9, f"beam_center 公式误差 {_w:.1e}"

ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v4w3"), "val")
consts = ds.norm_consts
# 注意：norm_consts 的 Hm/Wm 是探测器物理 extent（米）；像素数取自图像形状
Hm, Wm = ds._mimgs.shape[1], ds._mimgs.shape[2]
m1, dev, _ = load_px_nn(r"runs\v4s1\best.pt")
m1.eval()

# ---- 分箱选样（确定性）----
# 现实离轴子集（决定性对照）：低倾角(|rot|<0.02, 消除环≠poni 混杂+椭圆化)、
# λ∈[0.8,1.6]Å、画布内真环≥5（amp/DW 随机不可见线不计入）、束心在画布内
# （离轴≤0.45×边长）。1031 解剖：中心误差重尾(58px)×密集梳齿(Al2O3/Quartz)
# 会造出自洽伪解(pscore 0.61/m8/rms1.45)，281 号样本——本子集排除该机制，
# 让"NN 初始化 vs 几何中心"在现实条件下干净对决。
picks = {b: [] for b in BINS}
rng = np.random.default_rng(7)
order = rng.permutation(len(ds))
for i in order:
    g = int(ds._sel[i])
    p = ds._mparams[g]
    aux = ds._maux[g]
    if aux[1] < 4:                       # n_vis 环数门（生成端）
        continue
    if (abs(p[3]) > 0.02 or abs(p[4]) > 0.02
            or not (0.8 <= aux[5] <= 1.6)):
        continue
    ph = PHASE_NAMES[int(aux[0])]
    Q0 = phase_qs(ph)
    IR0 = PHASES[ph][1]
    k0 = IR0 >= 0.10
    rm0 = _r_model(Q0[k0], float(aux[5]), float(p[0]) / PX)
    if ((rm0 > 25) & (rm0 < min(Hm, Wm) * 0.707)).sum() < 5:
        continue                         # 画布内可配对真环不足
    cx_gt, cy_gt = beam_center_px(p[0], p[1], p[2], p[3], p[4])
    off = max(abs(cx_gt - (Wm - 1) / 2) / Wm, abs(cy_gt - (Hm - 1) / 2) / Hm)
    for b in BINS:
        if b[0] <= off < b[1] and len(picks[b]) < N_PER_BIN:
            picks[b].append(i)
            break
    if all(len(v) >= N_PER_BIN for v in picks.values()):
        break

print(f"== 离轴对决[现实子集]：val 域 "
      f"{sum(len(v) for v in picks.values())} 样本 "
      f"(每箱≤{N_PER_BIN}) ==")
print(f"{'箱(离轴)':>12}{'n':>4}{'A成功':>7}{'A中位ΔD%':>10}"
      f"{'B成功':>7}{'B中位ΔD%':>10}{'B中心err':>9}"
      f"{'C成功':>7}{'C中位ΔD%':>10}{'C中心err':>9}")


def run_arm(img, cx0, cy0, D_px0, Q, IR, lamA, D_ref):
    # 噪声域适配（1031 附录§六）：①bin≈1px（0.18px 过采样在泊松噪声下
    # 炸出 30+ 假峰）②必须传 prof_ctx——全谱相关分是伪解主杀手，
    # 漏传等于让覆盖分独扛（其对 amp/DW 随机不可见性系统性冤枉真解）
    nb = max(256, int(1.05 * float(np.hypot(*img.shape)) / 2))
    peaks, prof, bin_px, good = ring_radii(img, cx0, cy0, nb=nb)
    sol = solve(peaks, cx0, cy0, D_px0, Q, lamA, PX,
                I_rel=IR, canvas=img, prof_ctx=(prof, bin_px, good),
                D_ref_m=D_ref,
                d_range=(0.05, 3.0) if D_ref == 0 else (0.35, 2.8))
    if sol is None:
        return None
    return sol


for b in BINS:
    idxs = picks[b]
    if not idxs:
        continue
    rA, rB, rC = [], [], []
    cB, cC = [], []
    t0 = time.time()
    for i in idxs:
        g = int(ds._sel[i])
        img = np.asarray(ds._mimgs[g]).astype(np.float64)
        p = ds._mparams[g].astype(np.float64)
        aux = ds._maux[g]
        ph = PHASE_NAMES[int(aux[0])]
        Q = phase_qs(ph)
        IR = PHASES[ph][1]
        keep = IR >= 0.10          # 弱线滤除（与 ceria Q[:8] 同理：不可见线
        Q, IR = Q[keep], IR[keep]  # 会以 visible−m 罚真解的覆盖分）
        lamA = float(aux[5])
        D_gt = float(p[0])
        cx_gt, cy_gt = beam_center_px(p[0], p[1], p[2], p[3], p[4])

        # ---- A: 纯 NN（六参直出，参照）----
        x = torch.from_numpy(
            np.log1p(img[None, None].astype(np.float32))).to(dev)
        with torch.no_grad():
            mu, _ = m1(x, torch.tensor([lamA], device=dev),
                       torch.tensor([1.0], device=dev))
        p6 = norm_to_param(mu.double().cpu().numpy(), consts, 1.0)[0]
        rA.append(abs(p6[0] - D_gt) / D_gt * 100)
        cxN, cyN = beam_center_px(p6[0], p6[1], p6[2], p6[3], p6[4])

        # ---- B: 几何中心 + 全域 ----
        sB = run_arm(img, (Wm - 1) / 2, (Hm - 1) / 2, 1.0 / PX, Q, IR,
                     lamA, 0.0)
        if sB is None or sB[3] < 3:
            rB.append(np.nan)
            cB.append(np.nan)
        else:
            rB.append(abs(sB[2] * PX - D_gt) / D_gt * 100)
            cB.append(np.hypot(sB[0] - cx_gt, sB[1] - cy_gt))

        # ---- C: NN 六参 → 环心(倾角耦合修正) + 盆域门 ----
        sC = run_arm(img, cxN, cyN, p6[0] / PX, Q, IR, lamA, float(p6[0]))
        if sC is None or sC[3] < 3:
            rC.append(np.nan)
            cC.append(np.nan)
        else:
            rC.append(abs(sC[2] * PX - D_gt) / D_gt * 100)
            cC.append(np.hypot(sC[0] - cx_gt, sC[1] - cy_gt))

    rA = np.array(rA)
    rB = np.array(rB)
    rC = np.array(rC)
    cB = np.array(cB)
    cC = np.array(cC)
    okA = float(np.mean(rA < 1))
    okB = float(np.mean(rB < 1))     # NaN<1 = False → 失败计入分母
    okC = float(np.mean(rC < 1))
    lab = f"{b[0]:.2f}~{b[1]:.2f}"
    print(f"{lab:>12}{len(idxs):>4}"
          f"{okA*100:6.0f}%{np.median(rA):10.2f}"
          f"{okB*100:6.0f}%{np.nanmedian(rB):10.2f}{np.nanmedian(cB):9.1f}"
          f"{okC*100:6.0f}%{np.nanmedian(rC):10.2f}{np.nanmedian(cC):9.1f}"
          f"   [{time.time()-t0:.0f}s]")
print("\n注: 成功=|ΔD|/D<1%（B/C 另需环≥3）；A=NN直出, B=几何中心, C=NN+经典")
