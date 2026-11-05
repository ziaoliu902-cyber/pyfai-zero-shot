# -*- coding: utf-8 -*-
"""三波长真实对拍 v3：λ-conditioned 模型（base_v3lam）NN 起点 vs 名义起点。

R6 主实验：一个模型（0.5-2.3Å 训练）在三个真实波长上恢复指路能力：
  A. ceria 七档 (CeO2, Pilatus6M, λ=0.9724Å)   —— v1/v2 的老考卷
  B. AgBh 单档 (Pilatus1M, λ=1.000Å)            —— 探测器同款，最干净考卷
对照组（v1 复刻协议）：名义起点 = label dist + 探测器中心 + 零 tilt。
协议：单轨纯物理精修 400 iters（sigma_norm=None；v2 已证多起点扫描可解 250 假谷，
本实验只问：NN 起点是否一步进真谷）。
"""
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


def load_nn():
    ck = torch.load(os.path.join(ROOT, "runs/base_v3lam/best.pt"),
                    map_location="cpu", weights_only=False)
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


def run_case(name, ref, img, gt, starts):
    """starts: dict 起点名 -> 起点参数(6,)。返回各起点精修结果。"""
    px = ref.db.pixel
    print(f"\n[{name}] GT dist={gt[0]*1000:.1f}mm  poni=({gt[1]/px:.0f},{gt[2]/px:.0f}px)")
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
    return out


def main():
    m, dev, consts = load_nn()
    Hm, Wm = None, None

    # ================= 考卷 A：ceria 七档 =================
    ref = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3"),
                                detector="Pilatus6M", wavelength_m=LAM_CERIA,
                                dist_range=[0.1, 1.0])
    Hm, Wm = ref.db.extent
    A = {}
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        p_nn, sig = nn_coarse(m, dev, consts, img, 0.972386)
        nominal = np.array([gt[0], Hm / 2, Wm / 2, 0.0, 0.0, 0.0])
        A[d] = run_case(f"ceria {d}mm", ref, img, gt,
                        dict(NN=p_nn, nominal=nominal))
    for tag in ("NN", "nominal"):
        e = np.array([A[d][tag][2] for d in A])
        print(f"\nceria {tag:8s}: " + " ".join(f"{v:+.2f}" for v in e)
              + f"  中位|Δ|={np.median(np.abs(e)):.2f}mm")

    # ================= 考卷 B：AgBh @ Pilatus1M =================
    ref2 = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3"),
                                 detector="Pilatus_1M", wavelength_m=1.000e-10,
                                 dist_range=[0.1, 3.0])
    img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
    p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    H2, W2 = ref2.db.extent
    p_nn, sig = nn_coarse(m, dev, consts, img, 1.0)
    B = run_case("AgBh 1M", ref2, img, gt,
                 dict(NN=p_nn,
                      nominal=np.array([gt[0], H2 / 2, W2 / 2, 0.0, 0.0, 0.0])))
    print("\nλ 模型三波长之一（AgBh@1M）：NN 起点即真值谷" if abs(B["NN"][2]) < 3
          else "\nAgBh@1M：NN 起点未直接进谷，看名义对照")


if __name__ == "__main__":
    main()
