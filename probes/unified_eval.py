# -*- coding: utf-8 -*-
"""统一常数评测：多模型同卷（train_v3 test 4800）配对对比，常数错配效应显式化。

规矩（docs/对话记录_2026-10-26.md 第一节，2026-10-26 事故教训）：
  - 模型输出 mu_n 必须用【模型自己训练卷】meta.json 的常数 norm_to_param 解码
  - 真值 y_n 用【评测卷】常数解码（所有模型共用同一真值）
  - 拿评测卷常数去解别卷训练的模型 = 1311mm 假象（本探针附带复现这一列）

用法：
  python probes/unified_eval.py \
      --models "base_v3:runs/base_v3/last.pt:data/train_v3" \
               "base_v4b:runs/base_v4b/best.pt:data/finetune_v1" \
               "base_v4c:runs/base_v4c/best.pt:data/finetune_v2c"
可选重复传 --models。输出 probes/results/unified_eval_<ts>.md + .npz
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.dataset import TorchFAIDataset, norm_to_param, _norm_consts
from torchfai.model import ResNet18Geo
from torchfai.simulator import get_detector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "probes", "results")


@torch.no_grad()
def infer_mu(ckpt_fp, ds, batch=64):
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    m = ResNet18Geo()
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev).eval()
    from torch.utils.data import DataLoader
    mu, lv = [], []
    for x, y, aux in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
        x = x.unsqueeze(1).to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            mm, ll = m(x)
        mu.append(mm.float().cpu().numpy())
        lv.append(ll.float().cpu().numpy())
    return (np.concatenate(mu), np.concatenate(lv), ck.get("epoch"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="append", required=True,
                    help="名字:ckpt相对路径:训练卷相对路径（可多次）")
    ap.add_argument("--eval-data", default="data/train_v3")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    t0 = time.time()

    ds = TorchFAIDataset(os.path.join(ROOT, a.eval_data), split=a.split)
    consts_eval = ds.norm_consts
    # 真值：直接从打包件取（快）
    g = ds._sel
    y_p_true = ds._mparams[g].astype(np.float64)          # 物理量本身就是真值
    aux = ds._maux[g]
    dist_t = y_p_true[:, 0]
    n_vis = aux[:, 1].astype(int)

    # 像素尺寸（poni mm→px 换算；全部卷同探测器同 binning，取评测卷的）
    db = get_detector(ds.cfg["detector_name"], ds.cfg["binning"])
    px_mm = float(db.pixel) * 1000.0                       # m→mm

    tiers = {
        "全体": np.ones(len(ds), bool),
        "实际工况": (n_vis >= 6) & (aux[:, 3] >= 10000) & (aux[:, 4] < 0),
        "远距>0.77m": dist_t > 0.77,
    }
    print(f"评测卷 {a.eval_data}/{a.split} N={len(ds)}  "
          f"tiers: " + " ".join(f"{k}={v.sum()}" for k, v in tiers.items()))
    print(f"像素 {px_mm*1000:.0f}μm → 1px = {px_mm:.4f}mm\n")

    rows = []
    store = {}
    for spec in a.models:
        name, ck_rel, data_rel = spec.split(":")
        fp = os.path.join(ROOT, ck_rel)
        if not os.path.isfile(fp):
            print(f"[skip] {name}: 缺 {ck_rel}")
            continue
        mu_n, lv_n, ep = infer_mu(fp, ds)
        cfg_m = TorchFAIDataset(os.path.join(ROOT, data_rel), split="train").cfg
        consts_m = _norm_consts(cfg_m)
        mu_p = norm_to_param(mu_n.astype(np.float64), consts_m)   # 正确：自己卷的常数
        mu_p_wrong = norm_to_param(mu_n.astype(np.float64), consts_eval)  # 错配列
        err = np.abs(mu_p - y_p_true) * 1000.0          # m→mm, rad→mrad
        err_w = np.abs(mu_p_wrong - y_p_true) * 1000.0
        store[name] = dict(mu_p=mu_p, err=err, epoch=ep)

        for tname, m in tiers.items():
            e = err[m]
            ew = err_w[m]
            rows.append(dict(
                model=name, ep=ep, tier=tname, N=int(m.sum()),
                dist_mae=e[:, 0].mean(), dist_med=np.median(e[:, 0]),
                hit05=(e[:, 0] < 0.5).mean() * 100,
                poni_px=((e[:, 1] + e[:, 2]) / 2).mean() / px_mm,
                rot1=e[:, 3].mean(), rot2=e[:, 4].mean(),
                wrong_dec=ew[:, 0].mean()))
        print(f"  推理完成 {name} (ep{ep})  {time.time()-t0:.0f}s")

    # ---- 输出 ----
    hdr = (f"{'模型':14s}{'ep':>3s} {'档':8s}{'N':>5s} {'distMAE':>8s}{'dist中位':>8s}"
           f"{'<0.5mm':>7s} {'poni_px':>8s}{'rot1':>7s}{'rot2':>7s}{'错配解码MAE':>12s}")
    lines = [f"# 统一常数评测  {time.strftime('%Y-%m-%d %H:%M')}",
             f"评测卷：{a.eval_data}/{a.split}（N={len(ds)}，真值=评测卷物理量）",
             "解码：各模型用【自己训练卷】常数；末列=故意用评测卷常数的错配效应",
             "", "```", hdr]
    for r in rows:
        lines.append(
            f"{r['model']:14s}{r['ep']:>3d} {r['tier']:8s}{r['N']:>5d} "
            f"{r['dist_mae']:>8.2f}{r['dist_med']:>8.2f}{r['hit05']:>6.1f}% "
            f"{r['poni_px']:>8.2f}{r['rot1']:>7.2f}{r['rot2']:>7.2f}{r['wrong_dec']:>12.1f}")
    lines += ["```", "",
              "单位：dist mm；poni px；rot mrad。错配解码列复现 2026-10-26 的 1311mm 假象机制。"]
    txt = "\n".join(lines)
    print("\n" + txt)
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"unified_eval_{ts}.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    np.savez_compressed(os.path.join(OUT, f"unified_eval_{ts}.npz"),
                        y_p=y_p_true, aux=aux, **{f"{k}_mu": v["mu_p"] for k, v in store.items()})
    print(f"\n落盘 → probes/results/unified_eval_{ts}.{{md,npz}}")


if __name__ == "__main__":
    main()
