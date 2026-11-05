# -*- coding: utf-8 -*-
"""精修诊断二：环混淆局部谷能否用"多起点 dist"救回？

对 diag 一中拆笼仍卡/软卡的近距样本，从 NN 粗解出发、dist 缩放 0.75/1.35
做两个额外起点（无笼），与基线/拆笼结果按 loss 择优（>1 nat 才换）。
  救回 = 最终 dist 误差 < 5mm（原基线卡住）
输出：控制台 + probes/_refine_diag2_out.txt
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
from torchfai.refine import DifferentiableRefiner
from torchfai.phases import PHASE_NAMES

torch.set_num_threads(8)
DATA = os.path.join(ROOT, "data", "train_v2")

import shutil, tempfile
TMP = tempfile.mkdtemp(prefix="diag2_")
ck_fp = os.path.join(TMP, "b2.pt")
shutil.copy2(os.path.join(ROOT, "runs", "base_v2", "best.pt"), ck_fp)
ck = torch.load(ck_fp, map_location="cpu", weights_only=False)
model = ResNet18Geo(); model.load_state_dict(ck["model"]); model.eval()

ds = TorchFAIDataset(DATA, "test")
consts = ds.norm_consts
refs: dict[str, DifferentiableRefiner] = {}

# diag 一中拆笼仍卡/软卡的近距样本（results.csv 的 k 索引）
CASES = [3, 2869, 856, 2052, 2980]

L = []
def emit(s=""):
    print(s)
    L.append(s)

emit("== 多起点实验：环混淆谷（无笼，dist 缩放起点）==")
emit(f"{'样本':14s} {'dist':>6s} {'粗解mm':>7s} {'基线mm':>8s} | "
     f"{'x0.75mm':>8s} {'loss':>10s} | {'x1.35mm':>8s} {'loss':>10s} | 结论")

for k in CASES:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    true_p = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None]
        mu, lv = model(x)
        pred = norm_to_param(mu[0].float().numpy().astype(np.float64), consts)
        sig = np.exp(0.5 * lv[0].float().numpy()).astype(np.float64)

    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, DATA, device="cpu")

    # 基线（复用 diag 一配置）
    rA = refs[ph].refine(img, pred, iters=400, sigma_norm=sig, trust_k=0.5)
    eA = abs(rA["params"][0] - true_p[0]) * 1000

    # 多起点：dist 缩放，其余参数沿用 NN 粗解，无笼
    rows = []
    for fac in (0.75, 1.35):
        p2 = pred.copy(); p2[0] *= fac
        r = refs[ph].refine(img, p2, iters=400, sigma_norm=None)
        rows.append((fac, abs(r["params"][0] - true_p[0]) * 1000, r["loss"], r))

    best_fac, best_e, best_loss, best_r = min(rows, key=lambda t: t[2])
    switched = best_loss < rA["loss"] - 1.0
    final_e = best_e if switched else eA
    concl = ("救回" if final_e < 5 and eA >= 5 else
             ("更优" if final_e < eA * 0.5 and eA < 5 else
              ("仍卡" if final_e >= 5 else "维持")))
    emit(f"#{k:04d}{ph[:6]:>7s} {true_p[0]:6.3f} {abs(pred[0]-true_p[0])*1000:7.2f} {eA:8.2f} | "
         + " | ".join(f"{e:8.2f} {lo:10.4f}" for _, e, lo, _ in rows)
         + f" | {concl}" + (f"(换x{best_fac})" if switched else "(保基线)"))

emit("\n规则：多起点 loss 比基线好 >1 nat 才采纳；救回=最终误差<5mm")
with open(os.path.join(ROOT, "probes", "_refine_diag2_out.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
shutil.rmtree(TMP, ignore_errors=True)
print("\nreport -> probes/_refine_diag2_out.txt")
