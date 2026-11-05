# -*- coding: utf-8 -*-
"""中途对台：base_v3(当前best) vs base_v2(best) 在 v3-val 上分段对比。

回答"这次训练有没有用"：train_v3 的设计目的就是补近距/AgBh远距两类卡住样本，
所以看分段的 dist MAE 与 P(|Δdist|>5mm)（refine_v2 的卡住判据阈值）。
  A      = v2 老样本（其中 ~92.6% 在 base_v2 训练集里 → v2 主场，仅供参考）
  A-clean = A 中 v2 也没训过的行（v2 val/test 落点，干净）
  B      = 近距补课 0.1-0.35m（两模型都没训过 → 干净主战场）
  C      = AgBh 远距补课 0.8-3.0m（两模型都没训过 → 干净主战场）
CPU 推理（8 线程），不干扰正在跑的 GPU 训练。v3 还剩 20 epoch，本结果是 v3 下限。
"""
import sys
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo

torch.set_num_threads(8)
V3 = os.path.join(ROOT, "data", "train_v3")

# ---- 检查点先复制再读（避免与训练进程写 best.pt 竞争）----
import shutil
import tempfile
TMP = tempfile.mkdtemp(prefix="h2h_")
CKS = {}
for tag in ("base_v3", "base_v2"):
    src = os.path.join(ROOT, "runs", tag, "best.pt")
    dst = os.path.join(TMP, f"{tag}.pt")
    shutil.copy2(src, dst)
    ck = torch.load(dst, map_location="cpu", weights_only=False)
    CKS[tag] = ck
    print(f"{tag}: best epoch {ck['epoch']}  best_val {ck['best_val']:.4f}")

# ---- 数据与分段 ----
ds = TorchFAIDataset(V3, split="val")
sel = ds._sel                                  # 全局行号
consts = ds.norm_consts
p65 = np.random.default_rng(0).permutation(65000)
v2_train = set(p65[:57200].tolist())

seg = {}
mA = sel < 65000
seg["A(主场)"] = sel[mA]
seg["A(干净)"] = np.array([g for g in sel[mA] if g not in v2_train])
seg["B 近距"] = sel[(sel >= 65000) & (sel < 77000)]
seg["C AgBh远"] = sel[sel >= 77000]

rng = np.random.default_rng(0)
picks = {}
for k, g in seg.items():
    picks[k] = np.sort(g) if len(g) <= 260 else np.sort(rng.choice(g, 260, replace=False))
    print(f"段 {k}: 共 {len(g)}，抽 {len(picks[k])}")

# ---- 推理 ----
@torch.no_grad()
def predict(ck, gsel):
    model = ResNet18Geo()
    model.load_state_dict(ck["model"])
    model.eval()
    out = []
    for s in range(0, len(gsel), 32):
        xs, ys = [], []
        for g in gsel[s:s + 32]:
            x, y, _ = ds[int(np.searchsorted(sel, g))]
            xs.append(x); ys.append(y)
        x = torch.stack(xs).unsqueeze(1)          # (B,1,H,W)
        mu, _ = model(x)
        out.append(mu.numpy())
    mu_n = np.concatenate(out)
    return norm_to_param(mu_n, consts)             # 物理单位

idx_of = {g: i for i, g in enumerate(sel)}

# 每段真值单独取（直接用 ds 索引）
truth = {}
for k, gsel in picks.items():
    truth[k] = norm_to_param(np.stack([ds[idx_of[g]][1].numpy() for g in gsel]), consts)

print("\n段对比（dist MAE mm / poni1 MAE mm / P(|Δdist|>5mm) 卡住率）")
print(f"{'段':12s} {'N':>4s} {'v2: MAE':>9s} {'P>5mm':>7s} {'poni':>7s}   "
      f"{'v3: MAE':>9s} {'P>5mm':>7s} {'poni':>7s}   {'MAE比':>7s} {'卡住比':>7s}")
for k in picks:
    gsel = picks[k]
    t = truth[k]
    res = {}
    for tag in ("base_v2", "base_v3"):
        mu_p = predict(CKS[tag], gsel)
        e = (mu_p[:, 0] - t[:, 0]) * 1000.0                      # mm
        res[tag] = (np.abs(e).mean(), (np.abs(e) > 5).mean() * 100,
                    np.abs(mu_p[:, 1] - t[:, 1]).mean() * 1000)
    m2, p2, po2 = res["base_v2"]
    m3, p3, po3 = res["base_v3"]
    print(f"{k:12s} {len(gsel):4d} {m2:9.2f} {p2:6.1f}% {po2:7.2f}   "
          f"{m3:9.2f} {p3:6.1f}% {po3:7.2f}   {m3/m2:7.2f}x {p3/max(p2,1e-9):7.2f}x")

shutil.rmtree(TMP, ignore_errors=True)
print("\u6ce8\uff1av3 \u4e3a epoch28 \u68c0\u67e5\u70b9\uff08\u8fd8\u5269 20 epoch\uff0c\u5f53\u524d\u7ed3\u679c=\u4e0b\u9650\uff09\uff1b"
      "B/C \u6bb5\u4e24\u6a21\u578b\u5747\u672a\u8bad\u7ec3\u8fc7\uff0c\u662f\u5e72\u51c0\u5bf9\u6bd4\uff1bA(\u4e3b\u573a) \u6bb5 92.6% \u884c\u5728 base_v2 \u8bad\u7ec3\u96c6\u5185\u3002")
