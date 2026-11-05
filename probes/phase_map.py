# -*- coding: utf-8 -*-
"""条件相图：零样本盲解在 曝光/倾角/环数(距离)/畸变 × λ已知/未知 下的可解边界。

产物：probes/results/phase_map_<ts>.md + phase_map_<ts>.png（5 面板）
用法：python -X utf8 probes/phase_map.py [--quick]
"""
import os
import sys
import time

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from torchfai.simulator import get_detector, synth_sample
from torchfai.config import SimConfig
from solver_core import to_canvas, solve_auto, solve_auto_lambda

OUT = os.path.join(ROOT, "probes", "results")
os.makedirs(OUT, exist_ok=True)

LAM = 0.5               # Å（短波长，保证 9 条 CeO2 环都在画布内）
PH = "CeO2"
DET = ("Pilatus_1M", 1)
QUICK = "--quick" in sys.argv
NSEED = 3 if QUICK else 5


def make_cfg(expo, tilt_deg=0.0):
    return SimConfig(
        detector_name=DET[0], binning=DET[1], wavelength=LAM * 1e-10,
        dist_range=(0.10, 3.0),
        poni_frac_range=(0.60, 0.60),
        rot1_range=(np.radians(tilt_deg), np.radians(tilt_deg)),
        rot2_range=(np.radians(tilt_deg), np.radians(tilt_deg)),
        rot3_range=(0.0, 0.0),
        peak_sigma_range=(0.004, 0.004),
        amp_jitter=(1.0, 1.0),
        dw_b_range=(0.0, 0.0),
        bg0_range=(0.0, 0.0),
        bg1_range=(0.0, 0.0),
        hump_prob=0.0,
        peak_counts_range=(expo, expo),
        peak_counts_loguniform=False,
        hot_rate_range=(0.0, 0.0),
        dead_rate_range=(0.0, 0.0),
        beamstop_prob=0.0,
        min_visible_peaks=2,
    )


def run_point(dist_mm, expo, tilt_deg=0.0, lam_known=True, seeds=NSEED,
              distort_k=0.0, correct=True):
    """返回 [ (dD_mm, dcenter_px, phase_ok, lam_hat), ... ]"""
    db = get_detector(*DET)
    Hm, Wm = db.extent
    geom = dict(dist=dist_mm * 1e-3, poni1=0.60 * Hm, poni2=0.55 * Wm,
                rot1=np.radians(tilt_deg), rot2=np.radians(tilt_deg), rot3=0.0)
    cfg = make_cfg(expo, tilt_deg)
    idx = list(__import__("torchfai.phases", fromlist=["PHASES"]).PHASES).index(PH)
    res = []
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        s = synth_sample(rng, cfg, db, phase_idx=idx, geom=geom)
        img = s["img"].astype(np.float32)
        gt = s["geom"]
        if distort_k:
            img = barrel(img, distort_k) if not correct else barrel_inv(img, distort_k)
        canvas, sc, _, _ = to_canvas(img)
        pxm = db.pixel * sc
        try:
            if lam_known:
                r = solve_auto(canvas, LAM, pxm, cascade=True, auto_center=True)
                if r is None:
                    res.append((np.nan, np.nan, False, np.nan)); continue
                ph, sol = r
                lam_hat = LAM
            else:
                r = solve_auto_lambda(canvas, pxm, auto_center=True)
                if r is None:
                    res.append((np.nan, np.nan, False, np.nan)); continue
                ph, sol, lam_hat = r
                lam_hat = float(lam_hat) * 1e10 if lam_hat < 1e-6 else float(lam_hat)
        except Exception:
            res.append((np.nan, np.nan, False, np.nan)); continue
        dD = sol[2] * pxm * 1000 - gt["dist"] * 1000
        gcx = gt["poni2"] / db.pixel - 0.5
        gcy = gt["poni1"] / db.pixel - 0.5
        dc = float(np.hypot(sol[0] * sc - gcx, sol[1] * sc - gcy))
        res.append((dD, dc, (ph == PH), lam_hat))
    return res


def barrel(img, k):
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    dy, dx = yy - cy, xx - cx
    rmax = np.hypot(cy, cx)
    f = 1.0 + k * ((dy ** 2 + dx ** 2) / rmax ** 2)
    return map_coordinates(img, [cy + dy * f, cx + dx * f], order=1)


def barrel_inv(img, k):
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    sy, sx = yy.copy(), xx.copy()
    for _ in range(3):
        dy, dx = sy - cy, sx - cx
        rmax = np.hypot(cy, cx)
        f = 1.0 + k * ((dy ** 2 + dx ** 2) / rmax ** 2)
        sy, sx = cy + dy / f, cx + dx / f
    return map_coordinates(img, [sy, sx], order=1)


def summarize(rows):
    if not rows:
        return dict(n=0)
    dD = np.array([abs(r[0]) for r in rows], float)
    dc = np.array([r[1] for r in rows], float)
    pok = np.array([r[2] for r in rows], bool)
    lh = np.array([r[3] for r in rows], float)
    ok = np.isfinite(dD) & np.isfinite(dc)
    return dict(n=len(rows), ok=ok,
                dD=np.nanmedian(dD[ok]) if ok.any() else np.nan,
                dc=np.nanmedian(dc[ok]) if ok.any() else np.nan,
                pfrac=float(pok.mean()),
                lam=np.nanmedian(lh[ok]) if ok.any() else np.nan)


def main():
    t0 = time.time()
    rows_md = ["# 条件相图（合成，CeO2/Pilatus1M binned 2，λ=1.5406Å）", ""]

    # ---- 1 曝光 ----
    expo_list = (30, 100, 300, 1000, 3000, 10000, 30000)
    r1 = []
    for c in expo_list:
        s = summarize(run_point(150, c))
        r1.append((c, s)); print(f"[expo {c}] dD={s['dD']:.2f} dc={s['dc']:.2f} "
                                 f"ph={s['pfrac']:.2f}", flush=True)
    # ---- 2 倾角 ----
    tilts = (0, 0.5, 1, 2, 4)
    r2 = []
    for t in tilts:
        s = summarize(run_point(150, 10000, tilt_deg=t))
        r2.append((t, s)); print(f"[tilt {t}°] dD={s['dD']:.2f} dc={s['dc']:.2f} "
                                 f"ph={s['pfrac']:.2f}", flush=True)
    # ---- 3 距离（环数/角域）× λ 已知/未知 ----
    Ds = (60, 90, 130, 180, 250, 350, 480, 650)
    r3k, r3u = [], []
    for d in Ds:
        sk = summarize(run_point(d, 10000, lam_known=True))
        su = summarize(run_point(d, 10000, lam_known=False))
        r3k.append((d, sk)); r3u.append((d, su))
        print(f"[D {d}mm] known dD={sk['dD']:.2f} ph={sk['pfrac']:.2f} | "
              f"unk dD={su['dD']:.2f} λ̂={su['lam']:.3f} ph={su['pfrac']:.2f}", flush=True)
    # ---- 4 畸变 ----
    r4 = []
    for k in (0.0, 0.005, 0.01, 0.02):
        raw = summarize(run_point(150, 10000, distort_k=k, correct=False)) if k else summarize(run_point(150, 10000))
        cor = summarize(run_point(150, 10000, distort_k=k, correct=True)) if k else raw
        r4.append((k, raw, cor))
        print(f"[distort {k*100:.1f}%] raw dD={raw['dD']:.2f} | corrected dD={cor['dD']:.2f}",
              flush=True)

    # ---- 保存表 ----
    rows_md += ["## 1 曝光（峰计数）", "| 峰计数 | ΔD(mm) | Δcenter(px) | 相正确率 |", "|---|---|---|---|"]
    for c, s in r1:
        rows_md.append(f"| {c} | {s['dD']:.2f} | {s['dc']:.2f} | {s['pfrac']*100:.0f}% |")
    rows_md += ["", "## 2 倾角（rot1=rot2）", "| 倾角(°) | ΔD(mm) | Δcenter(px) | 相正确率 |", "|---|---|---|---|"]
    for t, s in r2:
        rows_md.append(f"| {t} | {s['dD']:.2f} | {s['dc']:.2f} | {s['pfrac']*100:.0f}% |")
    rows_md += ["", "## 3 距离（环数→角域）λ 已知 vs 未知",
                "| D(mm) | ΔD 已知 | 相 已知 | ΔD 未知 | λ̂ 未知 | 相 未知 |", "|---|---|---|---|---|---|"]
    for (d, sk), (_, su) in zip(r3k, r3u):
        rows_md.append(f"| {d} | {sk['dD']:.2f} | {sk['pfrac']*100:.0f}% | {su['dD']:.2f} | "
                       f"{su['lam']:.3f} | {su['pfrac']*100:.0f}% |")
    rows_md += ["", "## 4 桶形畸变（raw vs 校正）", "| k | raw ΔD | 校正后 ΔD |", "|---|---|---|"]
    for k, raw, cor in r4:
        rows_md.append(f"| {k*100:.1f}% | {raw['dD']:.2f} | {cor['dD']:.2f} |")
    ts = time.strftime("%Y%m%d_%H%M")
    md = os.path.join(OUT, f"phase_map_{ts}.md")
    open(md, "w", encoding="utf-8").write("\n".join(rows_md) + "\n")

    # ---- 图 ----
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    ax[0].loglog([c for c, _ in r1], [s["dD"] for _, s in r1], "o-")
    ax[0].set_xlabel("peak counts"); ax[0].set_ylabel("|ΔD| mm"); ax[0].set_title("1 exposure")
    ax[1].plot([t for t, _ in r2], [s["dc"] for _, s in r2], "o-")
    ax[1].set_xlabel("tilt deg"); ax[1].set_ylabel("Δcenter px"); ax[1].set_title("2 tilt")
    ax[2].semilogx([d for d, _ in r3k], [sk["dD"] for _, sk in r3k], "o-", label="λ known")
    ax[2].semilogx([d for d, _ in r3u], [su["dD"] for _, su in r3u], "s--", label="λ unknown")
    ax[2].set_xlabel("D mm"); ax[2].set_ylabel("|ΔD| mm"); ax[2].legend(); ax[2].set_title("3 distance")
    ax[3].semilogx([d for d, _ in r3u], [su["lam"] for _, su in r3u], "o-")
    ax[3].axhline(LAM, color="k", ls=":"); ax[3].set_xlabel("D mm"); ax[3].set_ylabel("λ̂ Å")
    ax[3].set_title("3b λ̂ vs D")
    ax[4].plot([k * 100 for k, _, _ in r4], [raw["dD"] for _, raw, _ in r4], "o-", label="raw")
    ax[4].plot([k * 100 for k, _, _ in r4], [cor["dD"] for _, _, cor in r4], "s--", label="corrected")
    ax[4].set_xlabel("barrel k %"); ax[4].set_ylabel("|ΔD| mm"); ax[4].legend(); ax[4].set_title("4 distortion")
    ax[5].plot([c for c, _ in r1], [s["pfrac"] for _, s in r1], "o-", label="exposure")
    ax[5].plot([t * 1e3 for t, _ in r2], [s["pfrac"] for _, s in r2], "s-", label="tilt×1e3")
    ax[5].set_xscale("log"); ax[5].set_ylabel("phase-correct"); ax[5].legend(); ax[5].set_title("phase ID")
    fig.tight_layout()
    png = os.path.join(OUT, f"phase_map_{ts}.png")
    fig.savefig(png, dpi=120)

    print(f"\n完成 {time.time()-t0:.0f}s → {md}\n{png}")
    print("\n".join(rows_md))


if __name__ == "__main__":
    main()

