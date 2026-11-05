# -*- coding: utf-8 -*-
"""2D 相图：倾角 × 曝光（Δcenter / ΔD / 相正确率）+ 真实样条闭环测试。

样条部分：用 pyFAI Spline 对合成图做**正向**畸变，再用 solver_core.undistort 校正，
比较 clean / raw(distorted) / corrected 三种情况的解——闭环验证 undistort 路径。
"""
import os
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from phase_map import run_point, summarize
from torchfai.simulator import get_detector, synth_sample
from torchfai.config import SimConfig
from torchfai.phases import PHASES
from solver_core import to_canvas, solve_auto, undistort

OUT = os.path.join(ROOT, "probes", "results")
TILTS = (0.0, 0.5, 1.0, 2.0, 4.0)
EXPOS = (100, 1000, 10000)


def main():
    print("== 2D 相图：倾角 × 曝光 ==", flush=True)
    dc = np.full((len(TILTS), len(EXPOS)), np.nan)
    dD = np.full_like(dc, np.nan)
    pf = np.zeros_like(dc)
    for i, t in enumerate(TILTS):
        for j, e in enumerate(EXPOS):
            s = summarize(run_point(150, e, tilt_deg=t, seeds=3))
            dc[i, j] = s["dc"]; dD[i, j] = s["dD"]; pf[i, j] = s["pfrac"]
            print(f"tilt={t:>4}° expo={e:>6}: dc={s['dc']:6.1f} dD={s['dD']:6.2f} ph={s['pfrac']:.2f}",
                  flush=True)

    # ---- 真实样条闭环：合成平滑位移场 → Spline → 正向畸变 → undistort ----
    print("== 样条闭环 ==", flush=True)
    from pyFAI.spline import Spline
    db = get_detector("Pilatus_1M", 1)
    H, W = db.shape
    Hm, Wm = db.extent
    geom = dict(dist=0.15, poni1=0.60 * Hm, poni2=0.55 * Wm,
                rot1=0.0, rot2=0.0, rot3=0.0)
    cfg = SimConfig(detector_name="Pilatus_1M", binning=1, wavelength=0.5e-10,
                    poni_frac_range=(0.60, 0.60), rot1_range=(0, 0), rot2_range=(0, 0),
                    rot3_range=(0, 0), peak_sigma_range=(0.004, 0.004),
                    amp_jitter=(1, 1), dw_b_range=(0, 0), bg0_range=(0, 0), bg1_range=(0, 0),
                    hump_prob=0, peak_counts_range=(10000, 10000),
                    hot_rate_range=(0, 0), dead_rate_range=(0, 0), beamstop_prob=0.0)
    idx = list(PHASES).index("CeO2")
    s = synth_sample(np.random.default_rng(7), cfg, db, phase_idx=idx, geom=geom)
    clean = s["img"].astype(np.float64)

    # 位移场：径向桶形 1%（平滑，控制点尺度 ~1/32 画幅）
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    dy0, dx0 = yy - cy, xx - cx
    rmax = np.hypot(cy, cx)
    k = 0.01
    f = k * ((dy0 ** 2 + dx0 ** 2) / rmax ** 2)
    DX = dx0 * f
    DY = dy0 * f

    sp = Spline()
    sp.xDispArray = DX
    sp.yDispArray = DY
    sp.array2spline(smoothing=1e-3)
    print("spline built; knots:", getattr(sp, "splineX", None) is not None, flush=True)

    def warp(img, sign=+1.0):
        dxx = np.asarray(sp.splineFuncX(xx, yy), float)
        dyy = np.asarray(sp.splineFuncY(xx, yy), float)
        return map_coordinates(img, [yy + sign * dyy, xx + sign * dxx], order=1)

    raw = warp(clean, +1.0)          # 记录图 = 真图按位移场错位
    cor = undistort(raw.astype(np.float32), sp)

    def solve(img):
        canvas, sc, _, _ = to_canvas(img.astype(np.float32))
        r = solve_auto(canvas, 0.5, db.pixel * sc, cascade=True, auto_center=True)
        if r is None:
            return None
        ph, sol = r
        gcx = geom["poni2"] / db.pixel - 0.5
        gcy = geom["poni1"] / db.pixel - 0.5
        return dict(ph=ph, dD=sol[2] * db.pixel * sc * 1000 - geom["dist"] * 1000,
                    dc=float(np.hypot(sol[0] * sc - gcx, sol[1] * sc - gcy)))

    res = {k: solve(v) for k, v in (("clean", clean), ("raw", raw), ("corrected", cor))}
    for k, v in res.items():
        print(f"  {k:9s}: {v}", flush=True)

    # ---- 出图 ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    im0 = ax[0].imshow(dc, aspect="auto", origin="lower", cmap="inferno")
    ax[0].set_xticks(range(len(EXPOS))); ax[0].set_xticklabels(EXPOS)
    ax[0].set_yticks(range(len(TILTS))); ax[0].set_yticklabels(TILTS)
    ax[0].set_xlabel("peak counts"); ax[0].set_ylabel("tilt deg")
    ax[0].set_title("Δcenter (px)"); fig.colorbar(im0, ax=ax[0])
    for i in range(len(TILTS)):
        for j in range(len(EXPOS)):
            ax[0].text(j, i, f"{dc[i,j]:.1f}", ha="center", va="center",
                       color="w", fontsize=8)
    im1 = ax[1].imshow(dD, aspect="auto", origin="lower", cmap="viridis")
    ax[1].set_xticks(range(len(EXPOS))); ax[1].set_xticklabels(EXPOS)
    ax[1].set_yticks(range(len(TILTS))); ax[1].set_yticklabels(TILTS)
    ax[1].set_xlabel("peak counts"); ax[1].set_ylabel("tilt deg")
    ax[1].set_title("|ΔD| (mm)"); fig.colorbar(im1, ax=ax[1])
    for i in range(len(TILTS)):
        for j in range(len(EXPOS)):
            ax[1].text(j, i, f"{dD[i,j]:.2f}", ha="center", va="center",
                       color="w", fontsize=8)
    vals = [res[k]["dD"] if res[k] else np.nan for k in ("clean", "raw", "corrected")]
    ax[2].bar(range(3), vals, color=["#4c72b0", "#dd8452", "#55a868"])
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["clean", "raw", "corrected"])
    ax[2].set_ylabel("|ΔD| mm"); ax[2].set_title("spline close-loop")
    fig.tight_layout()
    ts = np.datetime64("now").astype(str).replace(":", "")[:13]
    png = os.path.join(OUT, f"phase_map2d_{ts}.png")
    fig.savefig(png, dpi=120)
    print("saved", png)


if __name__ == "__main__":
    main()

