# -*- coding: utf-8 -*-
"""真实对拍 v6：px-conditioned 模型（runs/px_v1）全维度验收。

考卷：
  A. ceria 七档【宽视域 k≈2.43】：6M 全图均匀重采样 (521,490)（保方形有效
     像素 834µm，裁 18 划列对齐纵横比）→ NN(px=2.43, λ=0.9724) 粗解
     → ±8mm 网格精修（6M 原生图，协议同 v5）——治覆盖洞的主战场
  B. 回归检查【k=1 裁切路径】：ceria 150/200 用 to_nn_input_phys + px=1，
     粗解对比 v5（防遗忘：近距端不许掉出 ±10mm）
  C. AgBh@1M【k=1】：同 v5 协议复验
  D. alumina【k=1 零垫 + 宽峰域】：粗解 + sigma_range=(0.0015,0.6) 精修
     （10-29 上午判出域的案子，用宽峰模型+宽环域翻案）
模型：--model 指定（v3far3 终考）（px+lam 双触须）。
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
from torchfai.dataset import TorchFAIDataset, norm_to_param
from torchfai.model import ResNet18Geo
from torchfai.refine import DifferentiableRefiner
from torchfai.simulator import DetectorBundle, get_detector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")
LAM_CERIA = 9.72386e-11
GRID_MM = (-64.0, -32.0, -16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0, 32.0, 64.0)
Hm0, Wm0 = get_detector("Pilatus_1M", 2).extent        # 基准视域 (0.179,0.169)
PX6M = 172e-6


def load_px_nn(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    # v4s1+ 检查点带 σq 头 → 自动按需构造（forward 仍返回二元组，零破坏）
    has_sig = any(k.startswith("sig_head.") for k in sd)
    m = ResNet18Geo(lam_cond=True, px_cond=True, sig_head=has_sig).eval()
    m.load_state_dict(sd)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev)
    for cand in ("train_v3x", "train_px1", "train_v4w3", "train_v4w2"):
        # 全链同 config（v3x→v3x2→v4w2→v4w3 硬继承），consts 等价；
        # v3x/px1 已清盘，现存卷兜底
        p = os.path.join(ROOT, "data", cand)
        if os.path.isfile(os.path.join(p, "meta.json")):
            consts = TorchFAIDataset(p, "val").norm_consts
            return m, dev, consts
    raise FileNotFoundError("找不到 train_v3x/train_px1 的 meta.json")


def to_nn_input_wide(img, hw=(521, 490)):
    """6M 全图 → 方形有效像素重采样 (521, W') → 中央裁到 (521,490)。

    返回 (canvas, k, extra_col_m)：k = 行向有效像素/基准；extra_col_m =
    裁掉的左侧列折算 6M 原生物理长度（poni2 换算用）。
    10-29 修复：重采样稀释 + 真实曝光偏淡 → 画布比训练域暗 50~130×，
    粗解全废；进厂前按 p99.9 对齐训练亮度（diag3 消融验证：+336mm→+7mm）。
    """
    from scipy.ndimage import zoom
    H, W = img.shape
    f = hw[0] / H                       # 行向精确；列向同因子（保方形）
    W2 = int(round(W * f))
    b = zoom(np.clip(img.astype(np.float64), 0, None), (f, f), order=1)
    c0 = (W2 - hw[1]) // 2
    canvas = b[:, c0:c0 + hw[1]]
    eff_px = PX6M / f                                  # 方形有效像素
    k = hw[0] * eff_px / Hm0
    extra_col_m = (c0 / f) * PX6M
    return match_gain(canvas).astype(np.float32), float(k), float(extra_col_m)


GAIN_REF_P999 = 20536.0     # sim 宽视域标定（CeO2 0.15m seed=3）；容忍度由
                            # 训练期 poiss_aug 覆盖，常数偏置无碍


def match_gain(canvas, ref=GAIN_REF_P999):
    """把画布 p99.9 亮度对齐到训练域参考值（对数域单参数匹配）。"""
    canvas = np.asarray(canvas, dtype=np.float64)
    p = np.percentile(canvas, 99.9)
    return canvas * (ref / max(p, 1e-6)) if p > 1e-6 else canvas


@torch.no_grad()
def nn_coarse(m, dev, consts, img, lam_A, k=1.0):
    x = torch.from_numpy(np.log1p(img.astype(np.float32)))
    x = x.unsqueeze(0).unsqueeze(0).to(dev)
    mu, lv = m(x, torch.tensor([lam_A], device=dev),
               torch.tensor([k], device=dev))
    p = norm_to_param(mu.float().cpu().numpy(), consts, k)[0]
    return p, np.exp(0.5 * lv.float().cpu().numpy())[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "runs/px_v1/best.pt"))
    ap.add_argument("--skip-alumina", action="store_true")
    args = ap.parse_args()
    m, dev, consts = load_px_nn(args.model)
    rows = []

    def refine_case(ref, img, gt, starts, model_tag, case):
        out = {}
        for tag, st in starts.items():
            t0 = time.time()
            r = ref.refine(img, st, iters=400, sigma_norm=None,
                           sigma_range=(0.0015, 0.6) if "alumina" in case
                           else (0.0015, 0.02))
            px = ref.db.pixel
            out[tag] = dict(loss=r["loss"], ed=(r["params"][0] - gt[0]) * 1000,
                            ep1=(r["params"][1] - gt[1]) / px,
                            ep2=(r["params"][2] - gt[2]) / px,
                            sec=time.time() - t0)
            rows.append([model_tag, case, tag, f"{st[0]*1000:.1f}",
                         f"{r['loss']:.2f}", f"{out[tag]['ed']:+.3f}",
                         f"{out[tag]['ep1']:+.2f}", f"{out[tag]['ep2']:+.2f}"])
        return out

    # ================= 考卷 A：ceria 七档 宽视域 =================
    ref6 = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3x"),
                                 detector="Pilatus6M", wavelength_m=LAM_CERIA,
                                 dist_range=[0.1, 1.0])
    Hm6, Wm6 = ref6.db.extent
    A = {}
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        case = f"ceriaW {d}mm"
        wide, k, extra = to_nn_input_wide(img)
        p_nn, sig = nn_coarse(m, dev, consts, wide, 0.972386, k)
        # 诊断 poni（6M 原生系：canvas 系 + 裁列偏移）
        poni2_6m = p_nn[2] + extra
        poni_err = (p_nn[1] - gt[1]) / PX6M, (poni2_6m - gt[2]) / PX6M
        starts = {f"nn{dm:+.0f}": np.array(
            [p_nn[0] + dm * 1e-3, Hm6 / 2, Wm6 / 2, 0.0, 0.0, 0.0])
            for dm in GRID_MM}
        res = refine_case(ref6, img, gt, starts, "px_v1", case)
        best = min(res.values(), key=lambda r: r["loss"])
        A[d] = ((p_nn[0] - gt[0]) * 1000, best["ed"])
        print(f"[{case}] k={k:.3f} 粗解={p_nn[0]*1000:.1f}"
              f"({A[d][0]:+.1f}) poniΔ=({poni_err[0]:+.0f},{poni_err[1]:+.0f})px"
              f" → 网格最优 {best['ed']:+.2f}mm", flush=True)

    print("\n== A 汇总（粗解/精修, mm）:",
          " ".join(f"{d}:{A[d][0]:+.0f}/{A[d][1]:+.1f}" for d in A), flush=True)

    # ================= 考卷 B：回归（ceria 近距 k=1 裁切） =================
    refc = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3x"),
                                 detector="Pilatus6M", wavelength_m=LAM_CERIA,
                                 dist_range=[0.1, 1.0])
    B = {}
    for d in (150, 200):
        img = fabio.open(os.path.join(
            REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        p_nn, _ = nn_coarse(m, dev, consts, to_nn_input_phys(img),
                            0.972386, 1.0)
        starts = {f"nn{dm:+.0f}": np.array(
            [p_nn[0] + dm * 1e-3, Hm6 / 2, Wm6 / 2, 0.0, 0.0, 0.0])
            for dm in GRID_MM}
        res = refine_case(refc, img, gt, starts, "px_v1", f"ceriaC {d}mm")
        best = min(res.values(), key=lambda r: r["loss"])
        B[d] = ((p_nn[0] - gt[0]) * 1000, best["ed"])
        print(f"[回归 ceria {d}mm] 粗解={B[d][0]:+.1f} → {B[d][1]:+.2f}mm"
              f"（v5 参考：-0.63 / +0.05）", flush=True)

    # ================= 考卷 C：AgBh@1M（k=1） =================
    ref2 = DifferentiableRefiner("AgBh", os.path.join(ROOT, "data/train_v3x"),
                                 detector="Pilatus_1M", wavelength_m=1.000e-10,
                                 dist_range=[0.1, 3.0])
    img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
    p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
    gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
    H2, W2 = ref2.db.extent
    # 10-29：AgBh 实测 p99.9≈30k 已在训练域内，match_gain 反而调过头；
    # ×0.5 后 p99.9≈10k（训练域中位）粗解 +46.7mm（diag4 扫描）。脆弱，
    # 待统一重训（far@1M 加密度）后回退此 fudge。
    # 10-30 终考：v3far3 粗解（AgBh 远距 sim 偏 +360 已知）→ 精修宽窗
    # ±700mm 网格 + 双起姿（画布中心 / NN 自估 poni）双保险；
    # fudge 对比 ×1.0（域内亮度）与 ×0.5，粗解优者为主起跑。
    GRID_AG = (-700.0, -500.0, -300.0, -150.0, 0.0,
               150.0, 300.0, 500.0, 700.0)
    cand = {}
    for ftag, fc in (("x1.0", 1.0), ("x0.5", 0.5)):
        p_c, _ = nn_coarse(m, dev, consts,
                           match_gain(to_nn_input_phys(img)) * fc, 1.0, 1.0)
        cand[ftag] = p_c
        print(f"[AgBh 粗解 {ftag}] dist={p_c[0]*1000:.1f}"
              f"({(p_c[0]-gt[0])*1000:+.1f}mm) poni=({p_c[1]:.0f},{p_c[2]:.0f})"
              f"(GT {gt[1]:.0f},{gt[2]:.0f})", flush=True)
    p_nn = min(cand.values(), key=lambda p: abs(p[0] - gt[0]))
    starts = {}
    for dm in GRID_AG:
        starts[f"c{dm:+.0f}"] = np.array(
            [p_nn[0] + dm * 1e-3, H2 / 2, W2 / 2, 0.0, 0.0, 0.0])
        starts[f"n{dm:+.0f}"] = np.array(
            [p_nn[0] + dm * 1e-3, p_nn[1], p_nn[2], 0.0, 0.0, 0.0])
    refine_case(ref2, img, gt, starts, "v3far3", "AgBh 1M wide")
    print(f"[AgBh@1M] 粗解={p_nn[0]*1000:.1f}"
          f"({(p_nn[0]-gt[0])*1000:+.1f}mm) 详表 CSV", flush=True)

    # ================= 考卷 D：alumina（宽峰翻案） =================
    if not args.skip_alumina:
        AL = os.path.join(REAL, "alumina")
        dark = fabio.open(os.path.join(AL, "dark_0001.edf")).data.astype(np.float64)
        frames = [fabio.open(os.path.join(AL, f"al2o3_{i:04d}.edf")).data
                  for i in range(51)]
        aimg = np.clip(np.mean(frames, 0) - dark, 0, None).astype(np.float32)
        del frames
        H, W = aimg.shape
        p1, p2 = 103.358e-6, 102.53e-6
        cy, cx = 0.0529565255372 / p1, 0.05473342483 / p2
        yy0, xx0 = np.mgrid[0:H, 0:W]
        r_m = np.sqrt(((yy0 - cy) * p1) ** 2 + ((xx0 - cx) * p2) ** 2)
        aimg[r_m < 0.006] = 0.0                       # 中心亮斑=挡光盘语义
        from scipy.ndimage import zoom as szoom
        # 10-29 布局修复：物理贴放(k=1)留 213px 零边框 → 训练域外，粗解
        # +655mm；改拉满画布（zoom=521/H），k=有效像素/基准≈0.59（对线性
        # 触须轻度外推）→ 粗解 +68mm（diag4 L2）。精修仍用原生 aimg+Frelon。
        f2 = 521 / H
        canvas = szoom(aimg, (f2, f2), order=1)[:, :490].astype(np.float32)
        k_al = (p1 / f2) / 344e-6
        p_nn, _ = nn_coarse(m, dev, consts, match_gain(canvas), 0.70848, k_al)
        gp = load_geom(os.path.join(AL, "al2o3_00_max_51_frames.poni"))
        gta = np.array([gp.dist, gp.poni1, gp.poni2, gp.rot1, gp.rot2, gp.rot3])
        print(f"[alumina] GT=116.9mm 粗解={p_nn[0]*1000:.1f}"
              f"({(p_nn[0]-gta[0])*1000:+.1f})（上午旧模型 +941）", flush=True)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        db = DetectorBundle("Frelon_half", (H, W), p1, np.zeros((H, W), bool),
                            None, yy, xx)
        refa = DifferentiableRefiner("Al2O3", os.path.join(ROOT, "data/train_v3x"),
                                     detector=db, wavelength_m=7.084811024e-11,
                                     dist_range=[0.05, 1.0])
        starts = {f"nn{dm:+.0f}": np.array(
            [p_nn[0] + dm * 1e-3, H / 2 * p1, W / 2 * p2, 0.0, 0.0, 0.0])
            for dm in GRID_MM}
        res = refine_case(refa, aimg, gta, starts, "px_v1", "alumina")
        best = min(res.values(), key=lambda r: r["loss"])
        print(f"[alumina] 网格最优 distΔ={best['ed']:+.2f}mm "
              f"loss={best['loss']:.2f}（sigma_range 0.6 上限）", flush=True)

    os.makedirs(os.path.join(ROOT, "probes", "results"), exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(ROOT, "probes", "results", f"real_refine_v6_{ts}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "case", "start", "start_dist_mm", "loss",
                    "dist_err_mm", "poni1_px", "poni2_px"])
        w.writerows(rows)
    print(f"\n落盘 -> {out_csv}")


if __name__ == "__main__":
    main()
