# -*- coding: utf-8 -*-
"""精修诊断：近距卡住的病根在精修器内部，还是在盆外/笼内？

三个对照（每组 iters=400，与 refine_v2 同预算，CPU 运行不扰训练）：
  A 复现基线：NN 起点 + 0.5σ 信任域（refine_v2 原配置）
  B 真值起点：GT 起点 + 无笼子 —— 损失面在真值处稳不稳、优化器会不会跑飞
  C 拆笼子  ：NN 起点 + 无笼子 —— 卡住样本解笼后能不能自救

判读：
  - B 若稳在真值（<1mm）→ 精修器没病，损失面在真值处是稳定盆
  - B 的 loss_init（真值几何暖机后拟合分）vs A 的 best loss：
    A < B → 优化器找到了"比真值拟合更好"的假几何 → 可辨识性问题
  - C 若救回 → 笼子（信任域）太紧是元凶，扩笼/自适应笼可治
  - C 若仍卡 → 局部谷真存在，扩笼危险，只能治粗解（v3 方向）或多起点

样本：近距卡住 6（灰区5-10mm 2个 + 重症10-50mm 4个）+ 近距正常对照 2 + 远距AgBh卡住 2
输出：控制台 + probes/_refine_diag_out.txt（UTF-8）
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

# ---------- 1. 读 refine_v2 结果，选样本 ----------
rows = []
with open(os.path.join(ROOT, "runs", "refine_v2", "results.csv"), encoding="utf-8") as f:
    head = f.readline().strip().split(",")
    for line in f:
        v = line.strip().split(",")
        if len(v) == len(head):
            rows.append(v)
a = np.array([[float(x) for x in r[2:]] for r in rows])
kk = np.array([int(r[0]) for r in rows])
ph_arr = np.array([r[1] for r in rows])
dist, eb, ea = a[:, 0], a[:, 1:7], a[:, 7:13]
seg = np.where(dist < 0.39, 0, np.where(dist < 0.77, 1, 2))

def pick(mask, n, tag):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    take = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
    return [(int(kk[i]), tag) for i in take]

samples = (pick((seg == 0) & (eb[:, 0] > 5) & (eb[:, 0] <= 10) & (ea[:, 0] > 5), 2, "near灰区")
           + pick((seg == 0) & (eb[:, 0] > 10) & (ea[:, 0] > 5), 4, "near重症")
           + pick((seg == 0) & (eb[:, 0] < 3) & (ea[:, 0] < 1), 2, "near对照")
           + pick((seg == 2) & (ea[:, 0] > 5), 2, "far卡住"))
print(f"选中 {len(samples)} 样本: " + ", ".join(f"{t}#{k}" for k, t in samples))

# ---------- 2. 模型 + 数据 ----------
import shutil, tempfile
TMP = tempfile.mkdtemp(prefix="diag_")
src = os.path.join(ROOT, "runs", "base_v2", "best.pt")
ck_fp = os.path.join(TMP, "b2.pt"); shutil.copy2(src, ck_fp)
ck = torch.load(ck_fp, map_location="cpu", weights_only=False)
model = ResNet18Geo(); model.load_state_dict(ck["model"]); model.eval()

ds = TorchFAIDataset(DATA, "test")
consts = ds.norm_consts
refs: dict[str, DifferentiableRefiner] = {}

L = []  # report lines
def emit(s=""):
    print(s)
    L.append(s)

emit("\n== 精修三对照诊断 ==")
hdr = (f"{'样本':16s} {'dist':>6s} {'粗解mm':>7s} {'笼半径mm':>8s} | "
       f"{'A基线mm':>8s} {'lossA':>9s} | {'B真值mm':>8s} {'lossB':>9s} | "
       f"{'C拆笼mm':>8s} {'lossC':>9s} | 判读")
emit(hdr)

for k, tag in samples:
    g = int(ds._sel[k])
    img = np.asarray(ds._mimgs[g])
    true_p = ds._mparams[g].astype(np.float64)
    ph = PHASE_NAMES[int(ds._maux[g][0])]
    with torch.no_grad():
        x = torch.from_numpy(np.log1p(img.astype(np.float32)))[None, None]
        mu, lv = model(x)
        pred = norm_to_param(mu[0].float().numpy().astype(np.float64), consts)
        sig = np.exp(0.5 * lv[0].float().numpy()).astype(np.float64)

    # 笼半径（0.5σ+0.01 归一化 → 该距离处的物理 mm）
    margin = 0.5 * sig[0] + 0.01
    cage_mm = true_p[0] * (10 ** (margin * (consts["dhi"] - consts["dlo"]) / 2) - 1) * 1000

    if ph not in refs:
        refs[ph] = DifferentiableRefiner(ph, DATA, device="cpu")

    rA = refs[ph].refine(img, pred, iters=400, sigma_norm=sig, trust_k=0.5)
    rB = refs[ph].refine(img, true_p, iters=400, sigma_norm=None)
    rC = refs[ph].refine(img, pred, iters=400, sigma_norm=None)

    eA = abs(rA["params"][0] - true_p[0]) * 1000
    eB = abs(rB["params"][0] - true_p[0]) * 1000
    eC = abs(rC["params"][0] - true_p[0]) * 1000
    eb_mm = abs(pred[0] - true_p[0]) * 1000
    gt_best = min(rB["loss"], rB["loss_init"])
    verdict = []
    if eB < 1.0:
        verdict.append("GT稳")
    else:
        verdict.append(f"GT跑偏{eB:.1f}")
    if rA["loss"] < gt_best - 1e-4:
        verdict.append("假谷更深")
    else:
        verdict.append("真值更优")
    if eC < 5.0 and eA > 5.0:
        verdict.append("拆笼救回")
    elif eA < 5.0:
        verdict.append("基线本通")
    else:
        verdict.append("拆笼仍卡")
    emit(f"{tag}#{k:04d}{ph[:5]:>6s} {true_p[0]:6.3f} {eb_mm:7.2f} {cage_mm:8.2f} | "
         f"{eA:8.2f} {rA['loss']:9.4f} | {eB:8.3f} {rB['loss']:9.4f} | "
         f"{eC:8.2f} {rC['loss']:9.4f} | {'/'.join(verdict)}")

emit("\n判读图例：GT稳=真值处盆稳定；假谷优益=A的loss比真值还低(可辨识性问题)；"
     "拆笼救回=信任域太紧；拆笼仍卡=局部谷真实存在")
with open(os.path.join(ROOT, "probes", "_refine_diag_out.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
shutil.rmtree(TMP, ignore_errors=True)
print("\nreport -> probes/_refine_diag_out.txt")
