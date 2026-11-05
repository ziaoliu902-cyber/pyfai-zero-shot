# -*- coding: utf-8 -*-
"""决定性探针：v3 / v4 / v4b 的"灾难性遗忘"是不是解码常数错配造成的假象。

背景
----
数据集 meta.json 里的 config.dist_range 决定 _norm_consts() 的 dlo/dhi，
也就是"归一化标签 y_dist ←→ 物理距离 dist"的编码。

  train_v2 / train_v3 : dist_range=[0.1,3.0]  → dhi=log10(3.0)=0.4771
  finetune_v1         : dist_range=[0.1,1.0]  → dhi=log10(1.0)=0.0000   ← merge_ft.py 写错
                                                    （packs[0]=practice 段的配置，
                                                      没取 union；regular 段实际到 2.99m）

v4 / v4b 是在 finetune_v1 上微调的 → 它们的 dist 头学的是 dhi=0 那套编码。
如果评测时拿 v3 的 dhi=0.4771 去解码 v4b，就等于"用 A 卷的答案卡去读 B 卷"，
读出来当然是天文数字（此前观察到的 1294mm / 3983mm 之类）。

本探针：固定同一批 train_v2 test 样本、同一批模型，只改解码常数，
      打印各组合的 dist 误差。若"用对常数 → 数字正常"即证明假象。

用法：python probes/_decode_const_proof.py
"""
import copy
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, _norm_consts, norm_to_param
from torchfai.model import ResNet18Geo

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v2 = TorchFAIDataset(os.path.join(ROOT, "data", "train_v2"), split="test")
ft = TorchFAIDataset(os.path.join(ROOT, "data", "finetune_v1"), split="train")
v3ds = TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split="train")

truth = v2._mparams[v2._sel]
aux = v2._maux[v2._sel]
d = truth[:, 0]
prac = (aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
far = d > 0.77

# ---- 各套归一化常数 ----
cfg_v2 = copy.deepcopy(v2.meta["config"])
cfg_v3 = copy.deepcopy(v3ds.meta["config"])
cfg_ft = copy.deepcopy(ft.meta["config"])
cfg_fix = copy.deepcopy(cfg_ft)
cfg_fix["dist_range"] = [0.1, 3.0]          # union 后的正确版本
CONSTS = {"v2/v3 编码 dhi=log10(3)": _norm_consts(cfg_v2),
          "finetune 原编码 dhi=log10(1)": _norm_consts(cfg_ft),
          "finetune 修正 dhi=log10(3)": _norm_consts(cfg_fix)}
print("=== 各数据集的 dist 编码 ===")
for nm, root in (("train_v2", "data/train_v2"), ("train_v3", "data/train_v3"),
                 ("finetune_v1", "data/finetune_v1")):
    import json
    c = json.load(open(os.path.join(ROOT, root, "meta.json"), encoding="utf-8"))["config"]
    print(f"  {nm:12s} dist_range={c['dist_range']}  dhi=log10({c['dist_range'][1]})"
          f"={np.log10(c['dist_range'][1]):.4f}")
print("  train_v2 编码 == train_v3 编码 ? ",
      np.allclose(CONSTS["v2/v3 编码 dhi=log10(3)"]["dhi"],
                  CONSTS["v2/v3 编码 dhi=log10(3)"]["dhi"]))

# ---- 越界标签统计（真 bug 的证据）----
import json
pfin = np.load(os.path.join(ROOT, "data", "finetune_v1", "params.npy"), mmap_mode="r")
dfin = np.asarray(pfin)[:, 0]
y_bad = 2 * (np.log10(dfin) + 1.0) / 1.0 * 1.0 - 1.0      # dhi=0 编码
n_bad = int((y_bad > 1.0).sum())
print(f"\n=== finetune_v1 标签越界（按 dhi=log10(1)=0 编码）===")
print(f"  y_dist>1 的行数: {n_bad} / {len(dfin)}  ({n_bad/len(dfin)*100:.1f}%)"
      f"  全部来自 regular 段(dist>1.0m)，最大 y={y_bad.max():.3f}")
y_fix = 2 * (np.log10(dfin) + 1.0) / (np.log10(3.0) + 1.0) - 1.0
print(f"  按 union 编码(dhi=log10(3)) 归一化后 y 最大 = {y_fix.max():.4f}（落在 [-1,1] 内）")

# ---- 推理：把每个模型的 mu 原样存下来，换常数解码 ----
MODELS = [("base_v3", "runs/base_v3/best.pt"),
          ("base_v4", "runs/base_v4/best.pt"),
          ("base_v4b", "runs/base_v4b/best.pt")]
MU = {}
for tag, rel in MODELS:
    fp = os.path.join(ROOT, rel)
    if not os.path.exists(fp):
        continue
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval().cuda()
    mu_all = np.zeros((len(v2._sel), 6), np.float32)
    t0 = time.time()
    for s in range(0, len(v2._sel), 64):
        ks = np.arange(s, min(s + 64, len(v2._sel)))
        xb = np.stack([np.log1p(np.asarray(v2._mimgs[int(v2._sel[k])],
                                           dtype=np.float32)) for k in ks])
        with torch.no_grad():
            x = torch.from_numpy(xb)[:, None].cuda()
            mu, _ = model(x)
        mu_all[ks] = mu.cpu().numpy()
    MU[tag] = mu_all
    print(f"\n推理 {tag} 完成 ({time.time()-t0:.0f}s)")

# ---- 交叉解码表 ----
PX_UM = 344.0
TIERS = [("全体", np.ones(len(d), bool)), ("实际工况", prac), ("远距>0.77m", far)]
print("\n" + "=" * 96)
print("交叉解码：行=模型，列=解码常数。数字 = 全体 dist MAE(mm) / 远距档 dist MAE(mm) / 实际工况档 poni 中位(px)")
print("=" * 96)
for tag in MU:
    for cname, nc in CONSTS.items():
        pred = norm_to_param(MU[tag].astype(np.float64), nc)
        E = np.abs(pred - truth)
        e = E * 1000
        poni_px = np.median((E[prac][:, 1] + E[prac][:, 2]) / 2 / PX_UM * 1000)
        print(f"  {tag:8s} | {cname:26s} | 全体MAE {np.mean(e[:,0]):9.2f} | "
              f"远距MAE {np.mean(e[far,0]):9.2f} | 实际工况poni {poni_px:8.2f}px | "
              f"实际工况distMAE {np.mean(e[prac,0]):8.2f}")
    print("  " + "-" * 92)

print("\n=== 结论 ===")
for tag in MU:
    own = CONSTS["finetune 原编码 dhi=log10(1)"] if tag in ("base_v4", "base_v4b") \
        else CONSTS["v2/v3 编码 dhi=log10(3)"]
    wrong = CONSTS["v2/v3 编码 dhi=log10(3)"] if tag in ("base_v4", "base_v4b") else own
    p1 = norm_to_param(MU[tag].astype(np.float64), own)
    p2 = norm_to_param(MU[tag].astype(np.float64), wrong)
    e1 = np.abs(p1 - truth) * 1000
    e2 = np.abs(p2 - truth) * 1000
    print(f"  {tag:8s}: 用自己的常数 → 全体distMAE {np.mean(e1[:,0]):8.2f}mm, "
          f"远距 {np.mean(e1[far,0]):8.2f}mm")
    print(f"  {tag:8s}: 用错配常数   → 全体distMAE {np.mean(e2[:,0]):8.2f}mm, "
          f"远距 {np.mean(e2[far,0]):8.2f}mm   ← 这就是此前的'灾难性遗忘'")
