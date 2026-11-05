# -*- coding: utf-8 -*-
"""gen_sigq_labels.py：从图像实测等效环宽 → data/train_v4w3/sigq.npy。

背景：synth 的每环 σq 独立采样且未记录在 aux——σq 标签只能从图上量。
量"无量纲相对环宽" w = FWHM_px / r_px（对环的 q 位置近似不变，
σq↔w 单调），正是 dist 头被带偏的那个视觉量。

输出：
  sigq.npy      (N,) float32  原始 w（未归一化；nan 已填全体中位数）
  sigq_meta.json {lo, hi, nan_filled}   log10 域归一化边界
"""
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from scipy.signal import find_peaks
from torchfai.simulator import get_detector

DATA = os.path.join(ROOT, "data", "train_v4w3")
NB = 2048             # 标定修正（10-31）：512 时 bin 宽 ~1.2px ≈ 窄环 FWHM，
                      # 测量地板 0.07 吞掉全部动态范围；2048 → 0.3px 可分辨
MIN_R = 25.0          # 忽略束流挡光/中心垃圾区（px）
TOPK = 4              # 每图最多取 4 个峰
LOG = os.path.join(ROOT, "probes", "results", "sigq_label_live.log")


class _Tee:
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


def rel_width(img, pr, pc):
    """径向谱 → 主峰相对宽度中位数（无有效峰返回 nan）。"""
    H, W = img.shape
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    r = np.hypot(yy - pr, xx - pc)
    rmax = float(r.max())
    idx = (r / rmax * (NB - 1)).astype(np.int32).ravel()
    v = img.astype(np.float64).ravel()
    cnt = np.bincount(idx, minlength=NB)
    prof = np.bincount(idx, weights=v, minlength=NB) / np.maximum(cnt, 1)
    pk, _ = find_peaks(prof, prominence=0.15 * prof.max(), distance=4)
    if len(pk) == 0:
        return np.nan
    top = pk[np.argsort(prof[pk])[-TOPK:]]
    ws = []
    bin_px = rmax / NB
    for p in top:
        r_px = (p + 0.5) * bin_px
        if r_px < MIN_R:
            continue
        half = prof[p] * 0.5
        i = p
        while i > 0 and prof[i] > half:
            i -= 1
        j = p
        while j < NB - 1 and prof[j] > half:
            j += 1
        ws.append((j - i) * bin_px / r_px)
    return float(np.median(ws)) if ws else np.nan


def main():
    t0 = time.time()
    params = np.load(os.path.join(DATA, "params.npy"))
    aux = np.load(os.path.join(DATA, "aux.npy"))
    imgs = np.load(os.path.join(DATA, "imgs.npy"), mmap_mode="r")
    N = len(params)
    db0 = get_detector("Pilatus_1M", 2)
    PX0 = float(db0.pixel)
    out = np.full(N, np.nan, np.float32)
    n_nan = 0
    for g in range(N):
        k = float(aux[g, 6]) if aux.shape[1] >= 7 else 1.0
        px_eff = PX0 * k
        pr = params[g, 1] / px_eff
        pc = params[g, 2] / px_eff
        w = rel_width(np.asarray(imgs[g]), pr, pc)
        out[g] = w
        if np.isnan(w):
            n_nan += 1
        if (g + 1) % 2000 == 0 or g + 1 == N:
            print(f"label {g+1}/{N}  nan={n_nan}  {time.time()-t0:.0f}s", flush=True)
    med = float(np.nanmedian(out))
    n_fill = int(np.isnan(out).sum())
    out[np.isnan(out)] = med
    lg = np.log10(out)
    lo = float(np.percentile(lg, 0.5)) - 0.1
    hi = float(np.percentile(lg, 99.5)) + 0.1
    np.save(os.path.join(DATA, "sigq.npy"), out)
    with open(os.path.join(DATA, "sigq_meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(lo=lo, hi=hi, nan_filled=n_fill, median=med), f, indent=1)
    print(f"完成：N={N} nan填充={n_fill} w中位={med:.4f} "
          f"log10域[{lo:.2f},{hi:.2f}]  总耗时{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout = _Tee(LOG)
    main()
