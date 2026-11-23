# -*- coding: utf-8 -*-
"""poni 验证工具（叠加检查）：给定 图像 + poni（可指定标样相），
检查 poni 预测的环是否压在图上实际的环上。

用法：
    python poni_overlay.py <image> <poni> [相名] [--out 输出.png]

做的事：
1. 自动对齐中心（试 4 种轴映射，取环最锐的一种，兼容 rot3=90° 的 poni）；
2. 逐环检查：预测半径 ±窗口 内是否真有峰；
3. 出图：原图 + 预测环（绿）+ 圆心（红十字）+ 指标。
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")

import fabio
from solver_core import auto_offset
from torchfai.phases import PHASES, phase_qs


def parse_poni(p):
    kv = {}
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"^(\w+):\s*(.+)$", line.strip())
        if m:
            kv[m.group(1)] = m.group(2)
        mc = re.match(r"^#\s*Calibrant:\s*(\S+)", line.strip())
        if mc:
            kv["Calibrant"] = mc.group(1)
    cfg = json.loads(kv["Detector_config"])
    px = float(cfg.get("pixel1", 1e-4))
    return dict(px=px, dist=float(kv["Distance"]),
                p1=float(kv["Poni1"]), p2=float(kv["Poni2"]),
                r3=float(kv.get("Rot3", 0)),
                lam=float(kv["Wavelength"]) * 1e10 if "Wavelength" in kv else None,
                cal=kv.get("Calibrant"))


def profile_at(img, cx, cy, rmax, nb=4000):
    H, W = img.shape
    yy, xx = np.indices((H, W))
    r = np.hypot(yy - cy, xx - cx)
    idx = (r / rmax * nb).astype(np.int64)
    ok = r < rmax
    cnt = np.bincount(idx[ok], minlength=nb)
    prof = np.bincount(idx[ok], weights=img[ok], minlength=nb) / np.maximum(cnt, 1)
    return prof


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    img_p, poni_p = sys.argv[1], sys.argv[2]
    phase, out_png = None, None
    a = sys.argv[3:]
    i = 0
    while i < len(a):
        if a[i] == "--out":
            out_png = a[i + 1]
            i += 2
        else:
            phase = a[i]
            i += 1

    gt = parse_poni(poni_p)
    phase = phase or gt.get("cal")
    print(f"poni: D={gt['dist']*1000:.2f} mm  px={gt['px']*1e6:.1f} µm  "
          f"rot3={np.degrees(gt['r3']):.1f}°  λ={gt['lam']}  相={phase}")
    if phase is None or phase not in PHASES:
        print("！poni 里没有标样名，请显式给相（如 LaB6 / CeO2 / AgBh）")
        return

    img = auto_offset(np.asarray(fabio.open(img_p).data).astype(np.float32))
    H, W = img.shape
    qs = np.asarray(phase_qs(phase), float)   # 注意：phasor 表里存的是 d，必须换 q=2π/d
    lam = gt["lam"]
    tt = 2 * np.arcsin(np.clip(lam * qs / (4 * np.pi), 0, 0.999))
    r_pred = (gt["dist"] / gt["px"]) * np.tan(tt)

    gx, gy = gt["p2"] / gt["px"] - 0.5, gt["p1"] / gt["px"] - 0.5
    cands = {"direct": (gx, gy), "swap": (gy, gx),
             "flipx": (W - 1 - gx, gy), "flipy": (gx, H - 1 - gy)}
    rmax = 0.5 * min(H, W)
    best = None
    for name, (cx, cy) in cands.items():
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        prof = profile_at(img, cx, cy, rmax)
        sc = float(prof.max()) / (float(np.median(prof)) + 1e-9)
        if best is None or sc > best[0]:
            best = (sc, name, cx, cy, prof)
    sc, conv, cx, cy, prof = best
    print(f"中心映射: {conv} ({cx:.1f}, {cy:.1f})  锐度={sc:.1f}")

    binp = rmax / (len(prof) - 1)
    # 逐环用"局部对比度"判有没有峰：环带均值 / 两侧背景中位数。
    # 不用全局显著度——直射束（1.4e6）会把全局阈值顶到天上，
    # 而 AgB 的环只有背景的几倍（实测 273 px 处 4600 vs 背景 ~300）。
    hits, devs = 0, []
    contrasts = []
    for r0 in r_pred:
        if not (60 < r0 < rmax - 10):
            continue
        j = int(round(r0 / binp))
        w = max(2, int(round(3.0 / binp)))          # ±3 px 环带
        g = max(w + 2, int(round(18.0 / binp)))     # ±18 px 背景
        lo, hi = j - w, j + w + 1
        b_lo, b_hi = max(0, j - g), min(len(prof), j + g + 1)
        if hi >= len(prof) or lo < 1:
            continue
        band = float(prof[lo:hi].max())
        bg_idx = np.r_[np.arange(b_lo, max(b_lo, j - w - 1)),
                       np.arange(min(len(prof), j + w + 2), b_hi)]
        if bg_idx.size < 5:
            continue
        bg = float(np.median(prof[bg_idx])) + 1e-9
        contrast = band / bg
        contrasts.append(contrast)
        # 峰位偏差：环带内最大值的位置
        k = lo + int(np.argmax(prof[lo:hi]))
        devs.append(abs(k * binp - r0))
        if contrast > 1.3:
            hits += 1
    ntot = int(((r_pred > 60) & (r_pred < rmax - 10)).sum())
    med = float(np.median(devs)) if devs else float("nan")
    cmed = float(np.median(contrasts)) if contrasts else float("nan")
    print(f"环检查: {hits}/{ntot} 条对比度>1.3；中位对比度 {cmed:.2f}；"
          f"中位 |Δr| = {med:.2f} px")
    frac = (hits / ntot) if ntot else 0.0
    verdict = ("✓ good overlap" if frac >= 0.8 else
               "~ partial overlap" if frac >= 0.5 else "✗ no overlap")
    print("判语:", verdict)

    if out_png:
        lo, hi = np.percentile(img, [1, 99.8])
        v = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.45
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(v, cmap="gray", origin="upper")
        th = np.linspace(0, 2 * np.pi, 720)
        for r0 in r_pred:
            if 20 < r0 < rmax:
                ax.plot(cx + r0 * np.cos(th), cy + r0 * np.sin(th),
                        lw=0.7, color="#39a275", alpha=0.9)
        ax.plot([cx], [cy], marker="+", ms=10, color="#e53e3e", mew=1.4)
        ax.set_title(f"{os.path.basename(img_p)}  +  {os.path.basename(poni_p)}\n"
                     f"{phase}: {hits}/{ntot} rings overlapped, "
                     f"median |dr| = {med:.1f} px — {verdict}",
                     fontsize=8)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        print("图:", out_png)


if __name__ == "__main__":
    main()
