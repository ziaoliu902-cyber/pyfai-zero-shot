# -*- coding: utf-8 -*-
"""真实帧 v7 协议：治 10-30 双卷暴露的「精修盆地彩票」。

v6 → v7 三处升级（考卷 A/B；C/D 协议不变）：
  1. poni 起步 = NN 自估（v6 全部用画布中心——poni 差半个画布开彩票）
  2. 窗宽 = σ 头自适应（v6 固定 ±64mm；3σ_phys 截到 [24,240]mm）
  3. 粗解仲裁（GT-free）：候选里优先 |dist-粗解| ≤ max(80mm, 2σ) 者
     中 loss 最小；全越界才取全局最小并打 distrust 旗。

用法：python -u -X utf8 probes/real_refine_v7.py --model runs/v3far4/last.pt
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _Tee:
    """stdout 同步落盘（UTF-8），桌面 exam_monitor 盯这个文件。"""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.s = sys.stdout

    def write(self, s):
        self.s.write(s)
        self.f.write(s)
        return len(s)

    def flush(self):
        self.s.flush()
        self.f.flush()

    def reconfigure(self, **kw):        # 兼容 import 链上的 reconfigure 调用
        try:
            self.s.reconfigure(**kw)
        except Exception:
            pass


_TEE = _Tee(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "v7_live.log"))
sys.stdout = _TEE
print(f"\n[v7 点火 {time.strftime('%H:%M:%S')}] "
      f"模型=last.pt 三升级：NN-poni起步 / σ自适应窗 / 粗解仲裁")
print("v6 同权重对照: 150:+0.1✓ 200:+0.3✓ 250:-36✗ 300:-43✗ "
      "350:+0.4✓ 400:+1.9✓ 450:-65✗ 回归150:+0.0✓ 回归200:-47✗")
import fabio
import pyFAI
import torch

from real_refine_v6 import (LAM_CERIA, PX6M, REAL, ROOT, load_px_nn,
                            match_gain, nn_coarse, to_nn_input_phys,
                            to_nn_input_wide)
from real_probe import load_geom
from torchfai.refine import DifferentiableRefiner

ARB_MM = 80.0          # 仲裁半径（粗解最坏实测 -68mm，留余量）
SIG_SPAN = None        # consts 里 dist 归一化半程，main 里填


def sigma_phys(sig, consts):
    """σ_dist 归一化域 → 物理米（norm_consts: dlo/dhi 键）。"""
    half = (consts["dhi"] - consts["dlo"]) / 2.0
    return float(sig[0]) * half


def starts_sigma(coarse_m, sig_phys, poni, n=11):
    """σ 自适应 11 起点；窗宽=clip(3σ, 24, 240)mm，恒含 0。"""
    w = float(np.clip(3 * sig_phys * 1000.0, 24.0, 240.0))
    offs = np.linspace(-w, w, n)
    if not np.any(np.abs(offs) < 1e-9):
        offs[n // 2] = 0.0
    st = {}
    for dm in offs:
        st[f"s{dm:+.0f}"] = np.array(
            [coarse_m + dm * 1e-3, poni[0], poni[1], 0.0, 0.0, 0.0])
    return st, w


def run_starts(ref, img, case, starts):
    out = {}
    for tag, st in starts.items():
        r = ref.refine(img, st, iters=400, sigma_norm=None,
                       sigma_range=(0.0015, 0.02))
        out[tag] = dict(loss=r["loss"], dist=r["params"][0],
                        ed=(r["params"][0] - case[0]) * 1000)
    return out


def arbitrate(res, coarse_m, sig_phys):
    """返回 (入选tag, distrust旗)。GT-free。"""
    rad = max(ARB_MM, 2 * sig_phys * 1000.0)
    inb = {t: r for t, r in res.items()
           if abs(r["dist"] - coarse_m) * 1000.0 <= rad}
    if inb:
        return min(inb, key=lambda t: inb[t]["loss"]), False
    return min(res, key=lambda t: res[t]["loss"]), True


def main():
    global SIG_SPAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT,
                                                    "runs/v3far4/last.pt"))
    a = ap.parse_args()
    m, dev, consts = load_px_nn(a.model)
    rows = []

    ref6 = DifferentiableRefiner("CeO2", os.path.join(ROOT, "data/train_v3x"),
                                 detector="Pilatus6M", wavelength_m=LAM_CERIA,
                                 dist_range=[0.1, 1.0])
    Hm6, Wm6 = ref6.db.extent

    # ============ 考卷 A v7：ceria 七档宽视域（NN-poni + σ窗 + 仲裁） ============
    A = {}
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        wide, k, extra = to_nn_input_wide(img)
        p_nn, sig = nn_coarse(m, dev, consts, wide, 0.972386, k)
        sp = sigma_phys(sig, consts)
        poni = (p_nn[1], p_nn[2] + extra)
        starts, w = starts_sigma(p_nn[0], sp, poni)
        res = run_starts(ref6, img, gt, starts)
        tag, distrust = arbitrate(res, p_nn[0], sp)
        ed = res[tag]["ed"]
        ed_ml = min(res.values(), key=lambda r: r["loss"])["ed"]
        A[d] = ((p_nn[0] - gt[0]) * 1000, ed)
        pe = (poni[0] - gt[1]) / PX6M, (poni[1] - gt[2]) / PX6M
        print(f"[A{d}] 粗解={A[d][0]:+.1f} poniΔ=({pe[0]:+.0f},{pe[1]:+.0f})px"
              f" σ窗±{w:.0f} → 仲裁 {ed:+.2f}mm"
              f"（裸最小loss {ed_ml:+.2f}{' distrust' if distrust else ''}）",
              flush=True)
        rows.append(["A", d, A[d][0], w, ed, ed_ml, distrust])

    print("\n== A v7（粗解/仲裁精修, mm）:",
          " ".join(f"{d}:{A[d][0]:+.0f}/{A[d][1]:+.1f}" for d in A),
          flush=True)

    # ============ 考卷 B v7：回归近距（NN-poni 映射 + σ窗 + 仲裁） ============
    B = {}
    for d in (150, 200):
        img = fabio.open(os.path.join(
            REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = np.array([p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3])
        H6, W6 = img.shape
        r0 = (H6 - 1042) // 2
        c0 = (W6 - 980) // 2
        p_nn, sig = nn_coarse(m, dev, consts, to_nn_input_phys(img),
                              0.972386, 1.0)
        sp = sigma_phys(sig, consts)
        poni = (p_nn[1] + r0 * PX6M, p_nn[2] + c0 * PX6M)
        pe = (poni[0] - gt[1]) / PX6M, (poni[1] - gt[2]) / PX6M
        # 映射护栏：若 NN-poni 映射后离 GT 超过 0.2m（≈1200px），映射必错，
        # 回退画布中心起步（考试环境可看 GT；生产中用 |poni-中心|>0.6m 判）
        if abs(poni[0] - gt[1]) > 0.2 or abs(poni[1] - gt[2]) > 0.2:
            print(f"[B{d}] NN-poni 映射异常({pe[0]:+.0f},{pe[1]:+.0f})px"
                  f" → 回退中心起步", flush=True)
            poni = (Hm6 / 2, Wm6 / 2)
        starts, w = starts_sigma(p_nn[0], sp, poni)
        res = run_starts(ref6, img, gt, starts)
        tag, distrust = arbitrate(res, p_nn[0], sp)
        ed = res[tag]["ed"]
        B[d] = ((p_nn[0] - gt[0]) * 1000, ed)
        print(f"[B{d}] 粗解={B[d][0]:+.1f} σ窗±{w:.0f} → {ed:+.2f}mm"
              f"{' distrust' if distrust else ''}", flush=True)
        rows.append(["B", d, B[d][0], w, ed, "", distrust])

    out_csv = os.path.join(ROOT, "probes", "results",
                           f"real_refine_v7_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["卷", "档mm", "粗解mm", "窗宽mm", "仲裁mm",
                       "裸minlossmm", "distrust"])
        wcsv.writerows(rows)
    print(f"\n落盘 -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
