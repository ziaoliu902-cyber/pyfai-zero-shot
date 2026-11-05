# -*- coding: utf-8 -*-
"""train_v3 全量审计（只读，不改动任何数据）。

背景：2026-10-25 深夜 k-vs-g 索引错教训（考卷号 vs 数据行号）之后的标准动作——
凡是"行对齐"必须全量或强抽样验证。本审计七道关：

  1 结构    分片编号连续 / 长度模式 / 打包数组形状 dtype / 段边界
  2 血缘    四段行块的相与 dist 约束（v1 50k 全相 / far 15k AgBh / near 12k 全相近距 /
            agbh_far 3k AgBh 远距），相分布 vs 各源 meta
  3 恒等    v2 前缀 65000 行：params/aux 全量逐位 == train_v2；imgs 抽样逐位
  4 对齐    分片 npz ↔ 打包数组逐位比对（覆盖全部边界分片）+ 硬链接亲缘
  5 标签    params 范围/有限性；归一化往返；aux 值域；
            ★ n_vis 全量重算：环数完全由 (params, phase, bs_r) 决定 → 8 万行逐一比对
  6 图像    ★ 挡光盘圆心 == poni（params↔imgs 行对齐的图像级证据）；径向环位抽查
  7 切分    三份互斥全覆盖；与 TorchFAIDataset 复算一致；补课样本分布；
            benchmark 口径实测（v3×v2 切分交叉表）

2026-10-26 首跑结论：数据本体全过（含 ★ 两项）；两处 FAIL 均为审计阈值误设
（qmax 物理下限≈0.17 非 0.3；补课分布应按随机涨落 |z|<3 判）。另发现 meta 的
benchmark_note 口径错误：v2/v3 切分同用 seed=0，numpy 同种子不同长度置换
Spearman≈0.48 强相关 → v3-test 块A行仅 5.5%（215/3914）曾入 base_v2 训练集，
远非注记所称"约88%主场"。v3-test 对 base_v2 亦近乎未见 → 同卷对比实为干净
对比（非保守口径）。meta.benchmark_note 已按实测修正。
教训：跨代数据集对比必须实测切分重叠，不能按切分比例想当然；要独立就换种子。

用法：python probes/audit_train_v3.py
"""
import sys
import os
import json
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

for _s in (sys.stdout, sys.stderr):          # pythonw 兼容（train.py 同款保护）
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from torchfai.simulator import get_detector, ring_radius_m
from torchfai.phases import PHASE_NAMES, phase_qi
from torchfai.dataset import TorchFAIDataset, param_to_norm, norm_to_param, _norm_consts

V3 = os.path.join(ROOT, "data", "train_v3")
V2 = os.path.join(ROOT, "data", "train_v2")
NEAR = os.path.join(ROOT, "data", "hard_patch_v1", "near_mix")
FAR = os.path.join(ROOT, "data", "hard_patch_v1", "agbh_far")

T0 = time.time()
RESULTS = []


def sec():
    return f"[{time.time() - T0:6.1f}s]"


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'[OK]' if ok else '[FAIL]'} {sec()} {name}" + (f"\n         {detail}" if detail else ""),
          flush=True)


def note(msg):
    print(f"     · {sec()} {msg}", flush=True)


meta = json.load(open(os.path.join(V3, "meta.json"), encoding="utf-8"))
m2 = json.load(open(os.path.join(V2, "meta.json"), encoding="utf-8"))
m1 = json.load(open(os.path.join(ROOT, "data", "train_v1", "meta.json"), encoding="utf-8"))
mn = json.load(open(os.path.join(NEAR, "meta.json"), encoding="utf-8"))

db = get_detector(meta["config"]["detector_name"], meta["config"]["binning"])
H, W = db.shape
Hm, Wm = db.extent
PIX = float(db.pixel)
LAM_A = float(meta["config"]["wavelength"]) * 1e10

print(f"=== train_v3 审计  N(meta)={meta['total']}  探测器 {db.name}×bin{meta['config']['binning']}"
      f" → {H}×{W} @ {PIX*1e6:.0f}µm ===\n", flush=True)

# ---------------------------------------------------------------- 1 结构
shards = sorted(f for f in os.listdir(V3) if f.startswith("shard_") and f.endswith(".npz"))
check("1a 分片编号 00000..%05d 连续无缺" % (len(shards) - 1),
      shards == [f"shard_{i:05d}.npz" for i in range(len(shards))], f"共 {len(shards)} 个")

lens = []
for fn in shards:
    with np.load(os.path.join(V3, fn)) as z:
        lens.append(int(len(z["params"])))
lens = np.array(lens)
offs = np.concatenate([[0], np.cumsum(lens)])
N = int(offs[-1])
check("1b 分片总数 == 80000 == meta.total", N == 80000 == meta["total"], f"N={N}")

# 段边界：v2 块 255 片(65000) | near 块 | agbh_far 块；单片 ≤ 256
b_ok = (offs[255] == 65000 and offs[302] == 77000 and lens.max() <= 256
        and 254 < len(shards))
check("1c 段边界 offsets[255]=65000 / offsets[302]=77000，片长≤256", b_ok,
      f"offs[255]={offs[255]} offs[302]={offs[302]} "
      f"片长分布 min={lens.min()} max={lens.max()}")

imgs = np.load(os.path.join(V3, "imgs.npy"), mmap_mode="r")
params = np.load(os.path.join(V3, "params.npy"))
aux = np.load(os.path.join(V3, "aux.npy"))
check("1d 打包数组形状/dtype",
      imgs.shape == (N, H, W) and str(imgs.dtype) == "uint16"
      and params.shape == (N, 6) and str(params.dtype) == "float32"
      and aux.shape == (N, 5) and str(aux.dtype) == "float32",
      f"imgs{imgs.shape}/{imgs.dtype} params{params.shape} aux{aux.shape}")

check("1e v3.config 与 v2.config 逐字一致（归一化空间不变）",
      meta["config"] == m2["config"])
check("1f phase_names == torchfai.PHASE_NAMES", list(meta["phase_names"]) == list(PHASE_NAMES))

# ---------------------------------------------------------------- 2 血缘四段
ph_i = aux[:, 0].astype(int)
nvis = aux[:, 1].astype(int)
dist = params[:, 0].astype(np.float64)
EPS = 1e-6


def cnt(a):
    return np.bincount(a, minlength=6)


def as_counts(d):
    return np.array([int(d.get(p, 0)) for p in PHASE_NAMES])


# A1: rows 0..49999 = train_v1（全相，dist 0.1-3.0）
c1 = cnt(ph_i[:50000])
check("2a A1[0:50000] 相分布 == train_v1 meta",
      np.array_equal(c1, as_counts(m1["counts"])), " ".join(f"{p}={v}" for p, v in zip(PHASE_NAMES, c1)))
check("2b A1 dist ∈ [0.1, 3.0]",
      dist[:50000].min() >= 0.1 - EPS and dist[:50000].max() <= 3.0 + EPS,
      f"min={dist[:50000].min():.4f} max={dist[:50000].max():.4f}")

# A2: rows 50000..64999 = far_patch_v1（AgBh 强制，dist 0.6-3.0）
a2m = m2["provenance"]["far_patch_v1"]
check("2c A2[50000:65000] 全部 AgBh 且 dist ∈ [0.6, 3.0]",
      (ph_i[50000:65000] == 5).all()
      and dist[50000:65000].min() >= 0.6 - EPS and dist[50000:65000].max() <= 3.0 + EPS,
      f"dist min={dist[50000:65000].min():.4f} max={dist[50000:65000].max():.4f}")

# B: rows 65000..76999 = near_mix（全相，dist 0.1-0.35）
cb = cnt(ph_i[65000:77000])
check("2d B[65000:77000] 相分布 == near_mix meta",
      np.array_equal(cb, as_counts(mn["counts"])), " ".join(f"{p}={v}" for p, v in zip(PHASE_NAMES, cb)))
check("2e B dist ∈ [0.1, 0.35]",
      dist[65000:77000].min() >= 0.1 - EPS and dist[65000:77000].max() <= 0.35 + EPS,
      f"min={dist[65000:77000].min():.4f} max={dist[65000:77000].max():.4f}")

# C: rows 77000..79999 = agbh_far（AgBh 强制，dist 0.8-3.0）
check("2f C[77000:80000] 全部 AgBh 且 dist ∈ [0.8, 3.0]",
      (ph_i[77000:] == 5).all()
      and dist[77000:].min() >= 0.8 - EPS and dist[77000:].max() <= 3.0 + EPS,
      f"dist min={dist[77000:].min():.4f} max={dist[77000:].max():.4f}")

check("2g 全量相分布 == v3 meta.counts",
      np.array_equal(cnt(ph_i), as_counts(meta["counts"])),
      " ".join(f"{p}={v}" for p, v in zip(PHASE_NAMES, cnt(ph_i))))

# ---------------------------------------------------------------- 3 v2 前缀恒等
p2 = np.load(os.path.join(V2, "params.npy"))
a2 = np.load(os.path.join(V2, "aux.npy"))
check("3a v2 前缀 params[:65000] 逐位恒等", np.array_equal(params[:65000], p2))
check("3b v2 前缀 aux[:65000] 逐位恒等", np.array_equal(aux[:65000], a2))

rng = np.random.default_rng(123)
im2 = np.load(os.path.join(V2, "imgs.npy"), mmap_mode="r")
rows_v2 = np.concatenate([rng.choice(65000, 24, replace=False), [0, 25555, 49999, 64999]])
ok_img = all(np.array_equal(np.asarray(imgs[i]), np.asarray(im2[i])) for i in rows_v2)
check("3c v2 前缀 imgs 抽样 27 行逐位恒等（含 v1/far 边界 49999）", ok_img)

# ---------------------------------------------------------------- 4 分片↔打包对齐
probe_shards = [0, 97, 195, 196, 254, 255, 300, 301, 302, 313]
bad = []
for si in probe_shards:
    o, n = int(offs[si]), int(lens[si])
    with np.load(os.path.join(V3, shards[si])) as z:
        if not (np.array_equal(z["imgs"], np.asarray(imgs[o:o + n]))
                and np.array_equal(z["params"], params[o:o + n])
                and np.array_equal(z["aux"], aux[o:o + n])):
            bad.append(si)
check("4a 分片 npz ↔ 打包数组逐位一致（10 片含全部边界）", not bad,
      f"抽查 {probe_shards}" + (f" 不一致={bad}" if bad else ""))

link_bad = 0
for si in range(255):
    if not os.path.samefile(os.path.join(V3, shards[si]), os.path.join(V2, shards[si])):
        link_bad += 1
for si in range(255, 302):
    if not os.path.samefile(os.path.join(V3, shards[si]),
                            os.path.join(NEAR, f"shard_{si-255:05d}.npz")):
        link_bad += 1
for si in range(302, 314):
    if not os.path.samefile(os.path.join(V3, shards[si]),
                            os.path.join(FAR, f"shard_{si-302:05d}.npz")):
        link_bad += 1
check("4b 314 分片硬链接亲缘正确（v2/near_mix/agbh_far）", link_bad == 0,
      f"异常 {link_bad} 个")

# ---------------------------------------------------------------- 5 标签与 aux 自洽
P = params.astype(np.float64)
fin = np.isfinite(P).all() and np.isfinite(aux).all()
check("5a params/aux 全部有限（无 NaN/Inf）", bool(fin))

cfg = meta["config"]
r_ok = ((P[:, 0] >= 0.1 - EPS) & (P[:, 0] <= 3.0 + EPS)
        & (P[:, 1] >= 0.06 * Hm - EPS) & (P[:, 1] <= 0.94 * Hm + EPS)
        & (P[:, 2] >= 0.06 * Wm - EPS) & (P[:, 2] <= 0.94 * Wm + EPS)
        & (np.abs(P[:, 3]) <= 0.07 + EPS) & (np.abs(P[:, 4]) <= 0.07 + EPS)
        & (np.abs(P[:, 5]) <= 0.17 + EPS))
check("5b 六参数全部落在 config 采样域内", bool(r_ok.all()),
      f"越界 {int((~r_ok).sum())} 行")

consts = _norm_consts(cfg)
Y = param_to_norm(P, consts)
P_rt = norm_to_param(Y, consts)
rel = np.abs(P_rt - P) / np.maximum(np.abs(P), 1e-12)
check("5c 归一化 ∈ [-1,1] 且往返复原（相对误差<1e-6）",
      float(np.abs(Y).max()) <= 1 + 1e-5 and float(rel.max()) < 1e-6,
      f"max|y|={np.abs(Y).max():.6f} 往返max rel={rel.max():.2e}")

bsr = aux[:, 4].astype(np.float64)
aux_ok = ((aux[:, 0] >= 0) & (aux[:, 0] <= 5) & (np.abs(aux[:, 0] - np.round(aux[:, 0])) < 1e-6)
          & (nvis >= 2)
          & (aux[:, 2] > 0.1) & (aux[:, 2] <= 100.0)
          & (aux[:, 3] > 0)
          & ((bsr == -1) | ((bsr >= 8.0 - EPS) & (bsr <= 45.0 + EPS))))
check("5d aux 值域：phase∈[0,5] 整数 / n_vis≥2 / qmax∈(0.1,100]（物理下限≈0.17@3m） / scale>0 / bs_r∈{-1}∪[8,45]",
      bool(aux_ok.all()), f"异常 {int((~aux_ok).sum())} 行")

# ★ n_vis 全量重算（与 synth_sample 同式：直到倾近似 + poni 角点 + 挡光盘阈值）
mism_rows, worst_margin = [], 0.0
for pi in range(6):
    idx = np.where(ph_i == pi)[0]
    qs, _ = phase_qi(PHASE_NAMES[pi])
    d = P[idx, 0][:, None]
    rr = ring_radius_m(qs[None, :], d, LAM_A)                      # (n,k)
    p1, p2v = P[idx, 1], P[idx, 2]
    r_far = np.hypot(np.maximum(p1, Hm - p1), np.maximum(p2v, Wm - p2v))
    thr = np.where(bsr[idx] > 0, (bsr[idx] + 3.0) * PIX, 0.0)
    vis = (rr > thr[:, None]) & (rr < 1.05 * r_far[:, None])
    nv = vis.sum(axis=1)
    diff = np.where(nv != nvis[idx])[0]
    for j in diff:
        g = int(idx[j])
        m_lo = float(np.abs(rr[j] - thr[j]).min() / max(thr[j], 1.0))
        m_hi = float(np.abs(rr[j] - 1.05 * r_far[j]).min() / (1.05 * r_far[j]))
        worst_margin = max(worst_margin, min(m_lo, m_hi))
        mism_rows.append(g)
n_mism = len(mism_rows)
check("5e ★ n_vis 重算 == aux[:,1]（8 万行全量；余量<1e-5 视为 float32 舍入）",
      n_mism == 0 or worst_margin < 1e-5,
      f"不一致 {n_mism} 行 / 80000，最差边界余量 {worst_margin:.2e}"
      + (f"，行号样例 {mism_rows[:8]}" if mism_rows else ""))

# ---------------------------------------------------------------- 6 图像↔标签对齐
# ★ 挡光盘：图像中以 (poni1,poni2)/pixel 为圆心、半径 bs_r 的圆盘必须全零
cand = np.where(bsr > 0)[0]
picks = []
for lo, hi in ((0, 50000), (50000, 65000), (65000, 77000), (77000, 80000)):
    cc = cand[(cand >= lo) & (cand < hi)]
    picks += list(rng.choice(cc, min(6, len(cc)), replace=False))
disk_bad = []
for i in picks:
    rc, cc0, R = P[i, 1] / PIX, P[i, 2] / PIX, bsr[i]
    m = (db.yy - rc) ** 2 + (db.xx - cc0) ** 2 <= R * R
    if int(np.asarray(imgs[i])[m].max()) != 0:
        disk_bad.append(i)
check("6a ★ 挡光盘圆心==poni 且盘内全零（24 行跨四段）", not disk_bad,
      f"异常行 {disk_bad}" if disk_bad else f"抽查 {len(picks)} 行全过")

# 径向环位抽查：预测可见环的环带均值应显著高于图像中位背景
ring_ratios = []
for i in rng.choice(N, 8, replace=False):
    img = np.asarray(imgs[i], dtype=np.float32)
    qs, _ = phase_qi(PHASE_NAMES[ph_i[i]])
    rr_pix = ring_radius_m(qs, P[i, 0], LAM_A) / PIX
    rc, cc0 = P[i, 1] / PIX, P[i, 2] / PIX
    rgrid = np.sqrt((db.yy - rc) ** 2 + (db.xx - cc0) ** 2)
    pos = img[img > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    best = 0.0
    for r in rr_pix:
        if r < bsr[i] + 4 or r > 0.85 * rgrid.max():
            continue
        m = np.abs(rgrid - r) <= 3
        if m.sum() < 80:
            continue
        best = max(best, float(img[m].mean()) / max(med, 1.0))
    ring_ratios.append(round(best, 2))
check("6b 径向环位抽查（预测环带/背景 ≥1.5 倍，8 行至少 6 过）",
      sum(r >= 1.5 for r in ring_ratios) >= 6, f"对比度 {ring_ratios}")

# ---------------------------------------------------------------- 7 切分
g0 = np.random.default_rng(0)
perm = g0.permutation(N)
n_tr, n_va = int(N * 0.88), int(N * 0.06)
tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
check("7a 切分规模 70400/4800/4800 且互斥全覆盖",
      (n_tr, n_va, N - n_tr - n_va) == (70400, 4800, 4800)
      and len(np.intersect1d(tr, va)) == 0 and len(np.intersect1d(tr, te)) == 0
      and len(np.intersect1d(va, te)) == 0 and len(np.union1d(tr, np.union1d(va, te))) == N)

ds_tr = TorchFAIDataset(V3, split="train")
check("7b 独立复算 train._sel 与 TorchFAIDataset 一致",
      np.array_equal(np.sort(tr), ds_tr._sel), f"len={len(ds_tr)}")

hard = np.arange(65000, N)
hv = np.array([int(np.intersect1d(hard, s).size) for s in (tr, va, te)])
ex = np.array([15000 * r for r in (0.88, 0.06, 0.06)])
sd = np.array([(15000 * r * (1 - r)) ** 0.5 for r in (0.88, 0.06, 0.06)])
check("7c 补课 15k 三切分按比例分布（|z|<3 随机涨落）",
      bool((np.abs(hv - ex) < 3 * sd).all()),
      f"train/val/test={hv.tolist()}，期望±σ="
      + "/".join(f"{e:.0f}±{s:.0f}" for e, s in zip(ex, sd)))

# benchmark 口径实测：v2 实际切分 = Dataset 默认（seed=0, 0.88/0.06/0.06, N=65000）
# ★ 发现（2026-10-26）：v2/v3 同用 seed=0，numpy 同种子不同长度置换 Spearman≈0.48
#   强相关 → v3-test 的 v2 块行系统性落在 v2 val/test，仅 5.5% 曾入 v2 训练集
#   （若两切分独立则应为 ~88%）。v3-test 对 base_v2 亦近乎未见 → 同卷对比干净公平。
p65 = np.random.default_rng(0).permutation(65000)
v2tr, v2va, v2te = p65[:57200], p65[57200:61100], p65[61100:]
s3 = np.zeros(N, np.int8)
for k, part in enumerate((tr, va, te)):
    s3[part] = k
s2 = np.zeros(65000, np.int8)
for k, part in enumerate((v2tr, v2va, v2te)):
    s2[part] = k
lab = ("train", "val", "test")
note("benchmark 实测：v3×v2 切分交叉表（块A 65000 行，行=v3 列=v2）")
for i in range(3):
    rowc = [int(((s3[:65000] == i) & (s2 == j)).sum()) for j in range(3)]
    note(f"  v3-{lab[i]:5s} " + " ".join(f"v2-{lab[j]}={c:5d}" for j, c in enumerate(rowc)))
teA = te[te < 65000]
home = int(np.intersect1d(teA, v2tr).size)
pct = home / max(len(teA), 1) * 100
note(f"  → v3-test 块A行 {len(teA)} 张仅 {home} 张（{pct:.1f}%）曾入 base_v2 训练集；"
     f"另 {len(te)-len(teA)} 张为 v3 新增补课（base_v2 必然未见）")
pos8 = np.empty(N, np.int64); pos8[perm] = np.arange(N)
pos65 = np.empty(65000, np.int64); pos65[p65] = np.arange(65000)
sp = np.corrcoef(pos8[:65000].astype(float), pos65.astype(float))[0, 1]
note(f"  同种子置换秩相关 Spearman={sp:.3f}（独立应≈0；换种子独立切分时主场占比≈88%）。"
     f"meta.benchmark_note 已按本实测修正")

n_b = [(int(((nvis >= lo) & (nvis <= hi)).sum()), f"{lo}-{hi if hi < 90 else '+'}环")
       for lo, hi in ((2, 2), (3, 3), (4, 5), (6, 99))]
note("可见环数分布：" + "，".join(f"{nm}={c}（{c/N*100:.1f}%）" for c, nm in n_b))
note(f"近距补课块B[65000:77000] 2环占比 "
     f"{(nvis[65000:77000]==2).mean()*100:.1f}%；AgBh远距块C 6+环占比 "
     f"{(nvis[77000:]>=6).mean()*100:.1f}%")

n_dup = N - len(np.unique(params, axis=0))
note(f"params 精确重复行：{n_dup}（连续域采样，预期≈0）")

# ---------------------------------------------------------------- 汇总
n_ok = sum(1 for _, ok in RESULTS if ok)
print("\n" + "=" * 62, flush=True)
print(f"审计完成：{n_ok}/{len(RESULTS)} 项通过，耗时 {time.time()-T0:.0f}s", flush=True)
for name, ok in RESULTS:
    if not ok:
        print(f"  [FAIL] {name}", flush=True)
print("结论：train_v3 数据可信，可开训" if n_ok == len(RESULTS)
      else "结论：存在未通过项，开训前先排查上方 [FAIL]", flush=True)
sys.exit(0 if n_ok == len(RESULTS) else 1)
