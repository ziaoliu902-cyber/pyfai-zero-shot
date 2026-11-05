# -*- coding: utf-8 -*-
"""真实对拍 v5（固化协议）：物理一致输入 + 密网格多起点。

相对 v4 的修复（2026-10-29 凌晨诊断链结论）：
  1. NN 输入换 to_nn_input_phys（不缩放不转置，环尺度=训练分布）
  2. NN 起点精修 = ±8mm 密网格 5 起点，取纯 loss 最优（盆半宽≈dist 3-5%，
     +6mm 粗解恰在盆缘外，密网格是近距端到端跑通的必要条件）
  3. 名义起点与模型无关，只算一次（省 7 次精修）

考卷：A = ceria 七档（Pilatus6M, λ=0.9724Å）；B = AgBh@Pilatus1M（λ=1.0Å）。
模型：base_v3lamp（主）/ base_v3lam（对照，poiss_aug 净效应）。
输出：probes/results/real_refine_v5_<model>_<ts>.csv（全起点留痕）。
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

from real_probe import load_geom, to_nn_input_phys
from real_refine_v4 import load_nn
from torchfai.dataset import norm_to_param
from torchfai.refine import DifferentiableRefiner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")
LAM_CERIA = 9.72386e-11
GRID_MM = (-8.0, -4.0, 0.0, 4.0, 8.0)


@torch.no_grad()
def nn_coarse(m, dev, consts, img, lam_A):
    x = torch.from_numpy(np.log1p(to_nn_input_phys(img).astype(np.float32)))
    x = x.unsqueeze(0).unsqueeze(0).to(dev)
    mu, lv = m(x, torch.tensor([lam_A], device=dev))
    p = norm_to_param(mu.float().cpu().numpy(), consts)[0]
    return p, np.exp(0.5 * lv.float().cpu().numpy())[0]


def refine_starts(ref, img, gt, starts, rows, model_tag, case):
    """starts: dict tag->物理参数。全部跑，返回 {tag: 结果dict}。"""
    px = ref.db.pixel
    out = {}
    for tag, st in starts.items():
        t0 = time.time()
        r = ref.refine(img, st, iters=400, sigma_norm=None)
        ed = (r["params"][0] - gt[0]) * 1000
        out[tag] = dict(loss=r["loss"], ed=ed,
                        ep1=(r["params"][1] - gt[1]) / px,
                        ep2=(r["params"][2] - gt[2]) / px,
                        sec=time.time() - t0)
        rows.append([model_tag, case, tag, f"{st[0]*1000:.1f}",
                     f"{r['loss']:.2f}", f"{ed:+.3f}",
                     f"{out[tag]['ep1']:+.2f}", f"{out[tag]['ep2']:+.2f}"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base_v3lamp,base_v3lam")
    ap.add_argument("--grid", default=",".join(str(g) for g in GRID_MM),
                    help="NN 起点密网格 mm 偏移，逗号分隔")
    args = ap.parse_args()
    grid = [float(g) for g in args.grid.split(",")]
    tags = [t.strip() for t in args.models.split(",")]

    models = {t: load_nn(f"runs/{t}/best.pt") for t in tags}
    rows = []

    # ================= 考卷 A：ceria 七档 =================
    ref = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3"),
                                detector="Pilatus6M", wavelength_m=LAM_CERIA,
                                dist_range=[0.1, 1.0])
    Hm, Wm = ref.db.extent
    A = {t: {} for t in tags}
    nominal_done = {}
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        case = f"ceria {d}mm"
        nominal_done = refine_starts(       # 名义与模型无关，只算一次/档
            ref, img, gt,
            {"nominal": np.array([gt[0], Hm / 2, Wm / 2, 0.0, 0.0, 0.0])},
            rows, "shared", case)
        for mt in tags:
            m, dev, consts = models[mt]
            p_nn, _ = nn_coarse(m, dev, consts, img, 0.972386)
            nn_err = (p_nn[0] - gt[0]) * 1000
            starts = {f"nn{dm:+.0f}": np.array(
                [p_nn[0] + dm * 1e-3, Hm / 2, Wm / 2, 0.0, 0.0, 0.0])
                for dm in grid}
            res = refine_starts(ref, img, gt, starts, rows, mt, case)
            best = min(res.values(), key=lambda r: r["loss"])
            A[mt][d] = (nn_err, best["ed"])
            print(f"[{case} {mt}] 粗解={p_nn[0]*1000:.1f}({nn_err:+.1f}) "
                  f"→ 网格最优 distΔ={best['ed']:+.2f}mm "
                  f"(名义 {nominal_done['nominal']['ed']:+.2f})", flush=True)

    # ================= 考卷 B：AgBh @ Pilatus1M =================
    ref2 = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3"),
                                 detector="Pilatus_1M", wavelength_m=1.000e-10,
                                 dist_range=[0.1, 3.0])
    img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
    p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    H2, W2 = ref2.db.extent
    for mt in tags:
        m, dev, consts = models[mt]
        p_nn, _ = nn_coarse(m, dev, consts, img, 1.0)
        refine_starts(ref2, img, gt,
                      {"nn": p_nn,
                       "nominal": np.array([gt[0], H2 / 2, W2 / 2, 0.0, 0.0, 0.0])},
                      rows, mt, "AgBh 1M")
        print(f"[AgBh@1M {mt}] 粗解={p_nn[0]*1000:.1f}"
              f"({(p_nn[0]-gt[0])*1000:+.1f}mm) 详表见 CSV", flush=True)

    # ================= 汇总 =================
    print("\n===== ceria 七档汇总（粗解 err / 网格最优 err, mm）=====")
    for mt in tags:
        es = " ".join(f"{d}:{A[mt][d][0]:+.0f}/{A[mt][d][1]:+.1f}"
                      for d in A[mt])
        med = np.median([abs(A[mt][d][1]) for d in A[mt]])
        print(f"  {mt:12s}: {es}   全档中位|Δ|={med:.2f}")
        near = [abs(A[mt][d][1]) for d in (150, 200)]
        print(f"               近距档(150/200) |Δ| = "
              + "/".join(f"{v:.2f}" for v in near))

    os.makedirs(os.path.join(ROOT, "probes", "results"), exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(ROOT, "probes", "results", f"real_refine_v5_{ts}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "case", "start", "start_dist_mm", "loss",
                    "dist_err_mm", "poni1_px", "poni2_px"])
        w.writerows(rows)
    print(f"\n落盘 -> {out_csv}")


if __name__ == "__main__":
    main()
