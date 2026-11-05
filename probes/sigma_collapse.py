# -*- coding: utf-8 -*-
"""σ 头塌缩诊断：σ 是不是只是 clamp 下限的一个常数？

起因（10-27 续记二）：tail_anatomy 里 NN 自报 σ(poni1) 的三分位表退化成
一组（394 张全是三分位 3），追查发现 **poni1 的 σ 有 25~50% 的样本完全相等**。
进一步查 σ 分布：

    dist   0.04979 0.04979 0.04979 0.04979 0.04979 0.04979 0.04979  CV=0.0000
    poni1  0.04979 0.04979 0.04979 0.04979 0.16383 0.40391 0.52471  CV=1.0456

而 model.py 里 `logvar.clamp(-6.0, 4.0)` → σ = exp(0.5*lv) ∈ [0.0498, 7.39]，
**0.04979 = exp(0.5 × -6.0) 正是 clamp 下限**。
即：dist 通道 σ **100% 卡在下限**（恒为常数），poni/rot 有 25~50% 也卡在下限。

假说：Gaussian NLL 头在训练集上过拟合 → 想要无限置信 → logvar 被推到 -∞
→ 被 clamp 兜在 -6 → 对 OOD/难样本（尤其远距 AgBh）"自信地错"且 σ 极小
→ 精修的 σ-信任域半径被压到极小 → **尾部样本逃不出笼子**，
这正是 poni p75/p90 在所有配方下纹丝不动的机制。

本探针（一次性，train_v3 val+test）：
  1) 各通道 σ 落在 clamp 下限的样本比例（整体 / 分相 / 分档）
  2) 卡下限 vs 未卡下限两组的 z 分布（|z| 中位、RMS）
  3) 训练集（train split）上的同样统计 → 验证"过拟合→要无限置信"假说
  4) clamp 前后的 logvar 原始值（是否大量 < -6 被截断）
  5) 推论：信任域半径下限该怎么设（给出数值建议）

用法：python probes/sigma_collapse.py
"""
import io
import json
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param, PARAM_NAMES
from torchfai.model import ResNet18Geo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAMP_LO, CLAMP_HI = -6.0, 4.0
SIG_FLOOR = float(np.exp(0.5 * CLAMP_LO))
SIG_CEIL = float(np.exp(0.5 * CLAMP_HI))


@torch.no_grad()
def infer_raw(ckpt_fp, ds, batch=64):
    """返回 mu、lv_clamped、lv_raw（clamp 前）、y、aux。"""
    import torch.nn.functional as F
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    m = ResNet18Geo()
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev).eval()
    from torch.utils.data import DataLoader
    MU, LV, LVR, Y, AUX = [], [], [], [], []
    for x, y, aux in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
        x = x.unsqueeze(1).to(dev, non_blocking=True)
        if x.shape[-2:] != (512, 512):
            x = F.interpolate(x.float(), size=(512, 512), mode="bilinear",
                              align_corners=False)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            f = m.stem(x)
            f = m.stage4(m.stage3(m.stage2(m.stage1(f))))
            f = m.gap(f).view(x.shape[0], -1)
            mu = m.head_mu(f)
            lv_raw = m.head_logvar(f)
        MU.append(mu.float().cpu().numpy())
        LV.append(lv_raw.float().cpu().numpy())
        LVR.append(lv_raw.float().cpu().numpy().clip(CLAMP_LO, CLAMP_HI))
        Y.append(y.numpy())
        AUX.append(aux.numpy())
    return (np.concatenate(MU), np.concatenate(LVR), np.concatenate(LV),
            np.concatenate(Y), np.concatenate(AUX))


def stats(sig, y, mu, tag, n_out=8):
    out = [f"\n### {tag}（N={len(sig)}）", "",
           "| 通道 | σ=下限占比 | σ中位 | σ p95 | 卡下限组 \\|z\\|中位 | 未卡组 \\|z\\|中位 |",
           "|---|---|---|---|---|---|"]
    for j, nm in enumerate(PARAM_NAMES):
        s = sig[:, j]
        at_floor = s <= SIG_FLOOR * (1 + 1e-6)
        z = (y[:, j] - mu[:, j]) / s
        az = np.abs(z)
        a = az[at_floor].mean() if at_floor.any() else float("nan")
        b = az[~at_floor].mean() if (~at_floor).any() else float("nan")
        out.append(f"| {nm} | {at_floor.mean()*100:.1f}% | {np.median(s):.5f} | "
                   f"{np.percentile(s,95):.5f} | "
                   f"{a:.3f} | {b:.3f} |")
    return "\n".join(out)


def main():
    t0 = time.time()
    ck = torch.load(os.path.join(ROOT, "runs/base_v3/last.pt"),
                    map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"])
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    res = {}
    RAW = {}
    for split in ("train", "test", "val"):
        ds = TorchFAIDataset(os.path.join(ROOT, "data/train_v3"), split=split)
        mu, lvc, lvr, y, aux = infer_raw(
            os.path.join(ROOT, "runs/base_v3/last.pt"), ds)
        sig = np.exp(0.5 * lvc)
        res[split] = (sig, y, mu, aux)
        RAW[split] = lvr
        print(f"{split}: N={len(sig)}  {time.time()-t0:.0f}s", flush=True)

    out = []
    W = out.append
    W("# σ 头塌缩诊断\n")
    W(f"- 模型：`torchfai/model.py` → `logvar.clamp({CLAMP_LO}, {CLAMP_HI})`，"
      f"σ = exp(0.5·logvar) ∈ [{SIG_FLOOR:.5f}, {SIG_CEIL:.3f}]")
    W(f"- **σ 下限 = {SIG_FLOOR:.5f} = exp(0.5 × {CLAMP_LO})**")
    W(f"- 检查点：runs/base_v3/last.pt\n")

    for split in ("test", "train", "val"):
        sig, y, mu, aux = res[split]
        W(stats(sig, y, mu, f"{split} 集"))

    # ---- 相别 × 卡下限率（test 实际工况档）----
    sig, y, mu, aux = res["test"]
    from torchfai.phases import PHASE_NAMES
    ph = PHASE_NAMES
    actual = (aux[:, 1] >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0)
    print(f"\n[test 实际工况档 N={int(actual.sum())}]")
    phase_ids = aux[:, 0].astype(int)
    W("\n### 分相 × σ 卡下限率（test 实际工况档）\n")
    W("| 相 | N | " + " | ".join(PARAM_NAMES) + " |")
    W("|---|---|" + "---|" * len(PARAM_NAMES))
    for pid in sorted(set(phase_ids[actual])):
        m = actual & (phase_ids == pid)
        cells = []
        for j in range(len(PARAM_NAMES)):
            f = (sig[m, j] <= SIG_FLOOR * (1 + 1e-6)).mean() * 100
            cells.append(f"{f:.0f}%")
        W(f"| {ph[pid]} | {int(m.sum())} | " + " | ".join(cells) + " |")
    W("")

    # ---- 关键推论：信任域半径下限 ----
    W("## 推论：σ-信任域需要一个**绝对下限**\n")
    sig, y, mu, aux = res["test"]
    m = actual
    W("当前信任域半径 = trust_k · σ（归一化空间）。σ 卡在 0.04979 时：\n")
    W("| 通道 | σ(卡下限) | trust_k=1 时归一化半径 | 物理含义 |")
    W("|---|---|---|---|")
    # 归一化 → 物理：dist 是 log10 尺度，poni 是 mm，rot 是 mrad
    import json as _json
    meta = _json.load(io.open(os.path.join(ROOT, "data/train_v3/meta.json"),
                              encoding="utf-8"))
    cfg = meta["config"]
    dlo, dhi = np.log10(cfg["dist_range"][0]), np.log10(cfg["dist_range"][1])
    W(f"| dist | {SIG_FLOOR:.5f} | ±{SIG_FLOOR:.5f} (log10 m) | "
      f"±{SIG_FLOOR:.5f} × (dhi-dlo={dhi-dlo:.3f}) → "
      f"±{10**((SIG_FLOOR*(dhi-dlo))):.4f} 倍距离 |")
    W(f"| poni1/2 | {SIG_FLOOR:.5f} | ±{SIG_FLOOR:.5f} (归一化) | "
      f"× 探测器半高/半宽 |")
    W(f"| rot1/2/3 | {SIG_FLOOR:.5f} | ±{SIG_FLOOR:.5f} (归一化) | "
      f"× rot 量程 |")
    W("")
    W("→ σ 卡下限的样本，其信任域半径是**固定常数**（不随样本难度变化），"
      "且在归一化空间极小 —— 对远距/低 q 样本不足以覆盖真值。\n")
    W("**建议**：σ-信任域半径改为 `max(trust_k · σ, r_min[channel])`，"
      "r_min 按通道的**物理最小可分辨量**设定（而非按 σ）；"
      "σ 只决定「比最小半径更宽时能走多远」。这比继续调 T 更根本。\n")

    # ---- clamp 前原始 logvar 分布（验证"想要无限置信"假说）----
    W("## clamp 前的原始 logvar（假说验证）\n")
    W("| 集 | 通道 | 原始 logvar p5 | p50 | p95 | < -6 占比 | 原始 < -20 占比 |")
    W("|---|---|---|---|---|---|---|")
    for split in ("train", "test", "val"):
        lvr = RAW[split]
        for j, nm in enumerate(PARAM_NAMES):
            v = lvr[:, j]
            W(f"| {split} | {nm} | {np.percentile(v,5):.3f} | "
              f"{np.percentile(v,50):.3f} | {np.percentile(v,95):.3f} | "
              f"{(v<CLAMP_LO).mean()*100:.1f}% | {(v<-20).mean()*100:.1f}% |")
    W("")
    W("→ 若原始 logvar 大量 < -6（甚至 < -20），说明 NLL 头在**训练集上学到"
      "「无限自信」，被 clamp 兜住；训练集与 test 集的比值差异则是过拟合程度的直接度量。\n")

    ts = time.strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(ROOT, "probes", "results", f"sigma_collapse_{ts}.md")
    io.open(fp, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))
    print(f"\n落盘 {fp}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
