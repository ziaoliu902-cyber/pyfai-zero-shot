# -*- coding: utf-8 -*-
"""真实对拍 v4：poiss_aug 验收卷（base_v3lamp vs base_v3lam 同协议对拍）。

协议与 v3 完全一致（单轨纯物理精修 iters=400、sigma_norm=None、
起点 = NN 粗解 vs 名义起点），仅两点扩展：
  1. --ckpt 参数化：同一考卷跑两个模型（lamp=验收组 / lam=对照组），差值即
     poiss_aug 的净效应
  2. 逐案例 CSV 落盘（v3 只打控制台），并补记 NN 粗解误差列——10-28 事故的
     病灶正是"粗解在真实低曝图上失去区分度"，验收要同时看粗解和精修两头

用法：
  python probes/real_refine_v4.py --quick                    # 冒烟：ceria 250mm
  python probes/real_refine_v4.py --ckpt runs/base_v3lamp/best.pt   # 验收组
  python probes/real_refine_v4.py --ckpt runs/base_v3lam/best.pt    # 对照组
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio
import pyFAI
import torch

from real_probe import load_geom, to_nn_input
from torchfai.refine import DifferentiableRefiner
from torchfai.model import ResNet18Geo
from torchfai.dataset import TorchFAIDataset, norm_to_param

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")
LAM_CERIA = 9.72386e-11


def load_nn(ckpt):
    path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ResNet18Geo(lam_cond=True)
    m.load_state_dict(ck["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev).eval()
    consts = TorchFAIDataset(os.path.join(ROOT, "data/train_lam_v1"),
                             split="test").norm_consts
    return m, dev, consts


@torch.no_grad()
def nn_coarse(m, dev, consts, img, lam_A):
    x = torch.from_numpy(np.log1p(to_nn_input(img).astype(np.float32)))
    x = x.unsqueeze(0).unsqueeze(0).to(dev)
    mu, lv = m(x, torch.tensor([lam_A], device=dev))
    p = norm_to_param(mu.float().cpu().numpy(), consts)[0]
    return p, np.exp(0.5 * lv.float().cpu().numpy())[0]


def run_case(name, ref, img, gt, starts, nn_err, rows, model_tag):
    px = ref.db.pixel
    print(f"\n[{name}] GT dist={gt[0]*1000:.1f}mm  poni=({gt[1]/px:.0f},{gt[2]/px:.0f}px)"
          f"  NN粗解误差={nn_err:+.1f}mm")
    out = {}
    for tag, st in starts.items():
        t0 = time.time()
        r = ref.refine(img, st, iters=400, sigma_norm=None)
        ed = (r["params"][0] - gt[0]) * 1000
        out[tag] = (st[0] * 1000, r["loss"], ed,
                    (r["params"][1] - gt[1]) / px, (r["params"][2] - gt[2]) / px,
                    time.time() - t0)
        print(f"  {tag:8s} 起点{st[0]*1000:7.1f}mm  loss={r['loss']:8.2f}  "
              f"distΔ={ed:+7.2f}mm  poniΔ=({out[tag][3]:+5.1f},{out[tag][4]:+5.1f})px  "
              f"{out[tag][5]:.0f}s", flush=True)
        rows.append([model_tag, name, tag, f"{st[0]*1000:.1f}", f"{nn_err:+.3f}",
                     f"{r['loss']:.2f}", f"{ed:+.3f}",
                     f"{out[tag][3]:+.2f}", f"{out[tag][4]:+.2f}", f"{out[tag][5]:.0f}"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base_v3lamp/best.pt")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    model_tag = os.path.basename(os.path.dirname(args.ckpt.rstrip("/\\"))) or "ckpt"

    m, dev, consts = load_nn(args.ckpt)
    print(f"模型 = {args.ckpt}  ({model_tag})  iters={args.iters}")
    rows = []

    # ================= 考卷 A：ceria 七档（quick 只考 250mm） =================
    ref = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3"),
                                detector="Pilatus6M", wavelength_m=LAM_CERIA,
                                dist_range=[0.1, 1.0])
    Hm, Wm = ref.db.extent
    dists = (250,) if args.quick else (150, 200, 250, 300, 350, 400, 450)
    A = {}
    for d in dists:
        img = fabio.open(os.path.join(
            REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        p_nn, sig = nn_coarse(m, dev, consts, img, 0.972386)
        nn_err = (p_nn[0] - gt[0]) * 1000
        nominal = np.array([gt[0], Hm / 2, Wm / 2, 0.0, 0.0, 0.0])
        A[d] = run_case(f"ceria {d}mm", ref, img, gt,
                        dict(NN=p_nn, nominal=nominal), nn_err, rows, model_tag)
    for tag in ("NN", "nominal"):
        e = np.array([A[d][tag][2] for d in A])
        print(f"\nceria {tag:8s}: " + " ".join(f"{v:+.2f}" for v in e)
              + f"  中位|Δ|={np.median(np.abs(e)):.2f}mm")

    # ================= 考卷 B：AgBh @ Pilatus1M（quick 跳过） =================
    if not args.quick:
        ref2 = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3"),
                                     detector="Pilatus_1M", wavelength_m=1.000e-10,
                                     dist_range=[0.1, 3.0])
        img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
        p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        H2, W2 = ref2.db.extent
        p_nn, sig = nn_coarse(m, dev, consts, img, 1.0)
        nn_err = (p_nn[0] - gt[0]) * 1000
        run_case("AgBh 1M", ref2, img, gt,
                 dict(NN=p_nn,
                      nominal=np.array([gt[0], H2 / 2, W2 / 2, 0.0, 0.0, 0.0])),
                 nn_err, rows, model_tag)

    os.makedirs(os.path.join(ROOT, "probes", "results"), exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(ROOT, "probes", "results",
                           f"real_refine_v4_{model_tag}_{ts}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "case", "start", "start_dist_mm", "nn_coarse_err_mm",
                    "refine_loss", "dist_err_mm", "poni1_px", "poni2_px", "sec"])
        w.writerows(rows)
    print(f"\n落盘 -> {out_csv}")


if __name__ == "__main__":
    main()
