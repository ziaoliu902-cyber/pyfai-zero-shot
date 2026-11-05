# -*- coding: utf-8 -*-
"""探测器域差诊断：6M→NN 管线里到底谁在杀粗解（2026-10-28 深夜）。

背景：real_refine_v4 验收发现 λ 模型在真实 ceria@6M 上远距过冲 ~2×。
to_nn_input 有三个捆绑嫌疑，本探针用【仿真图】拆开：
  ① 转置 bug：训练图 (521,490)，to_nn_input 输出 (490,521) —— 真实图
     从 v1 起就是转置着进 NN 的
  ② 非均匀缩放：6M (1263,1231)→(490,521)，行×0.388 列×0.423 —— 圆环变椭圆
  ③ poni 出域：真实 ceria poni 0.215m > 训练域 0.179/0.169m

五行 × 四距矩阵（全部仿真 CeO2, λ=0.9724Å, rot=0, 峰值~15000）：
  A native   : 训练网格(521,490)直接渲染 —— 对照组（=训练分布）
  B pipe     : 6M 渲染 → to_nn_input 原样 —— 真实管线复刻（①+②+③全占）
  C pipe_T   : B 转置回 (521,490) —— 去掉①，留②③
  D cropbin  : 6M 渲染取中央训练视域(1042×980) + 2×2 bin —— 去掉①②③
               （内容上等价于训练仿真器，物理一致性重采样的修法原型）
  E offT     : 6M 渲染 poni=训练域内(0.130,0.127m) → to_nn_input+转置 ——
               去掉①③，留②（与 C 对照拆 poni 出域）

判读：
  D≈A 好、B/C/E 坏  → 重采样语义（②）实锤 → 修法=D 同款物理重采样，免重训
  E 好而 C 坏       → poni 出域（③）主因 → 修法=NN 域扩容重训
  C 好而 B 坏       → 转置（①）主因 → 修法=管线加一行 .T
"""
import csv
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from real_probe import to_nn_input
from torchfai.refine import DifferentiableRefiner
from torchfai.model import ResNet18Geo
from torchfai.dataset import TorchFAIDataset, norm_to_param

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAM = 9.72386e-11
DISTS = (0.15, 0.25, 0.35, 0.45)
TARGET_PEAK = 15000.0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.default_rng(20261028)


def load_model(ckpt):
    ck = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    m = ResNet18Geo(lam_cond=True)
    m.load_state_dict(ck["model"])
    m.to(DEV).eval()
    return m


CONSTS = TorchFAIDataset(os.path.join(ROOT, "data/train_lam_v1"),
                         split="test").norm_consts


@torch.no_grad()
def render(ref, dist, poni1, poni2):
    """给定物理几何，用 refiner 的已验证物理渲染一张仿真图（float32 计数）。"""
    phys = np.array([dist, poni1, poni2, 0.0, 0.0, 0.0])
    pn = ref.phys_to_pn(phys, ref.consts)
    q = ref.q_map(pn.to(ref.device))
    amps = ref.Is_t * torch.exp(-0.01 * ref.qs_t * ref.qs_t)
    sc = torch.log(torch.tensor(TARGET_PEAK, device=ref.device) / amps.max())
    mu = ref._render(
        q, sc, torch.tensor(0.01, device=ref.device),
        torch.zeros(ref.n_ring, device=ref.device),
        torch.full((ref.n_ring,), float(np.log(0.007)), device=ref.device),
        torch.log(torch.tensor(0.5, device=ref.device)),
        torch.log(torch.tensor(0.05, device=ref.device)),
        torch.log(torch.tensor(0.5, device=ref.device)),
        torch.tensor(1.5, device=ref.device),
        torch.log(torch.tensor(0.25, device=ref.device)),
    )
    img = RNG.poisson(mu.float().cpu().numpy()).astype(np.float32)
    img[ref.dead.cpu().numpy()] = 0.0
    return img


@torch.no_grad()
def nn_pred(m, x):
    """x: (H,W) float 计数图 → (物理参数, σ_norm)"""
    t = torch.from_numpy(np.log1p(x.astype(np.float32)))
    t = t.unsqueeze(0).unsqueeze(0).to(DEV)
    mu, lv = m(t, torch.tensor([0.972386], device=DEV))
    p = norm_to_param(mu.float().cpu().numpy(), CONSTS)[0]
    return p, np.exp(0.5 * lv.float().cpu().numpy())[0]


def main():
    models = {t: load_model(f"runs/{t}/best.pt")
              for t in ("base_v3lamp", "base_v3lam")}
    px_tr = 344e-6  # 训练网格像素（172µm × binning2）

    ref_1m = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3"),
                                   wavelength_m=LAM)              # 训练网格
    ref_6m = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3"),
                                   detector="Pilatus6M", wavelength_m=LAM,
                                   dist_range=[0.1, 1.0])
    H6, W6 = ref_6m.db.shape
    Ht, Wt = ref_1m.db.shape
    ctr1, ctr2 = (H6 / 2) * 172e-6, (W6 / 2) * 172e-6   # 6M 中心（≈真实 ceria poni）
    off1, off2 = 0.3 * H6 * 172e-6, 0.3 * W6 * 172e-6   # 域内 poni（0.130/0.127m）

    print(f"训练网格 {ref_1m.db.shape} px={px_tr*1e6:.0f}µm extent="
          f"{ref_1m.db.extent[0]:.3f}x{ref_1m.db.extent[1]:.3f}m | "
          f"6M {ref_6m.db.shape} extent={ref_6m.db.extent[0]:.3f}x{ref_6m.db.extent[1]:.3f}m")
    print(f"中心 poni=({ctr1:.3f},{ctr2:.3f})m（出域）  偏心 poni=({off1:.3f},{off2:.3f})m（域内）\n")

    rows = []
    for d in DISTS:
        img_native = render(ref_1m, d, (Ht / 2) * px_tr, (Wt / 2) * px_tr)
        img_6m_c = render(ref_6m, d, ctr1, ctr2)
        img_6m_o = render(ref_6m, d, off1, off2)

        # D：中央裁剪训练视域 + 2×2 bin（物理一致重采样原型）
        r0 = H6 // 2 - 521, H6 // 2 + 521
        c0 = W6 // 2 - 490, W6 // 2 + 490
        crop = img_6m_c[r0[0]:r0[1], c0[0]:c0[1]]
        img_crop = crop.reshape(521, 2, 490, 2).mean((1, 3)).astype(np.float32)

        variants = {
            "A_native":   (img_native,           (Ht / 2) * px_tr, (Wt / 2) * px_tr),
            "B_pipe":     (to_nn_input(img_6m_c), ctr1, ctr2),
            "C_pipe_T":   (to_nn_input(img_6m_c).T, ctr1, ctr2),
            "D_cropbin":  (img_crop,             ctr1, ctr2),
            "E_offT":     (to_nn_input(img_6m_o).T, off1, off2),
        }
        for tag, (x, gp1, gp2) in variants.items():
            for mt, m in models.items():
                p, _ = nn_pred(m, x)
                ed = (p[0] - d) * 1000
                ep1 = (p[1] - gp1) / px_tr
                ep2 = (p[2] - gp2) / px_tr
                rows.append([mt, tag, f"{d*1000:.0f}", f"{p[0]*1000:.1f}",
                             f"{ed:+.1f}", f"{ep1:+.0f}", f"{ep2:+.0f}"])
                print(f"[{d*1000:.0f}mm {tag:9s} {mt}] 预测={p[0]*1000:6.1f}mm "
                      f"distΔ={ed:+7.1f}  poniΔ=({ep1:+5.0f},{ep2:+5.0f})px", flush=True)
        print()

    # ---------- 判读 ----------
    print("=" * 70)
    print("中位 |distΔ| (mm)，按 变体 × 模型：")
    med = {}
    for tag in ("A_native", "B_pipe", "C_pipe_T", "D_cropbin", "E_offT"):
        for mt in models:
            e = [abs(float(r[4])) for r in rows if r[1] == tag and r[0] == mt]
            med[(tag, mt)] = float(np.median(e))
            print(f"  {tag:9s} {mt:12s}: {med[(tag, mt)]:8.1f}")

    lamp = {t: med[(t, "base_v3lamp")] for t in
            ("A_native", "B_pipe", "C_pipe_T", "D_cropbin", "E_offT")}
    print("\n判读（base_v3lamp 口径，A 对照）：")
    if lamp["D_cropbin"] < 2 * max(lamp["A_native"], 5):
        print("  D≈A：物理一致重采样后 NN 正常 → 内容无罪，病在重采样/坐标系语义")
    else:
        print("  D 明显差于 A：连训练等价内容都失败 → 需再查（渲染/可见环数）")
    if lamp["C_pipe_T"] < 0.5 * lamp["B_pipe"]:
        print("  C 明显好于 B：转置 bug 是主因之一 → 修法=管线加 .T（一行）")
    if lamp["E_offT"] < 0.5 * lamp["C_pipe_T"]:
        print("  E 明显好于 C：poni 出域是主因之一 → 修法=NN 域扩容")
    elif lamp["C_pipe_T"] > 3 * max(lamp["D_cropbin"], lamp["A_native"], 5):
        print("  C 仍差（E 无改善）：非均匀缩放（圆环变形）实锤 → 修法=物理重采样")

    out = os.path.join(ROOT, "probes", "results",
                       f"resample_diag_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "true_dist_mm", "pred_dist_mm",
                    "dist_err_mm", "poni1_px", "poni2_px"])
        w.writerows(rows)
    print(f"\n落盘 -> {out}")


if __name__ == "__main__":
    main()
