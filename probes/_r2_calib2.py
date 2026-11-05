# -*- coding: utf-8 -*-
"""R2 门控校准探针 v2（18-1104 需求，精简版）。

只测生产中真正会发生的扫描：
  · Eiger4M / Eiger9M：s=5/7 → c-risk 同 λ 单点大画布（必须 Δps<0.05 拒收）
  · mar3450：默认档（应判弱）+ 大画布单点（看 b/c 哪个门捕获）
  · Frelon2k / halfccd：无 λ 联合默认档（b 门不得误伤）+ Frelon 大档
  · P2M / P1M / ceria：默认档统计（s<5 或锁表 → 生产不扫）
输出无缓冲落盘 results/_r2_calib2.txt。
"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

import fabio  # noqa: E402
import pyFAI  # noqa: E402

from solver_core import (solve_auto, solve_auto_lambda, to_canvas,  # noqa: E402
                         solve_cascade)
from torchfai.phases import phase_qs  # noqa: E402

ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")


def load_img(path):
    if path.endswith(".h5"):
        import h5py
        with h5py.File(path, "r") as f:
            hit = []

            def _walk(name, obj):
                if not hit and isinstance(obj, h5py.Dataset) \
                        and obj.ndim >= 2 and name.split("/")[-1] == "data":
                    hit.append(obj)
            f.visititems(_walk)
            arr = hit[0][()]
            img = np.asarray(arr[0] if arr.ndim == 3 else arr, dtype=np.float64)
    else:
        img = np.asarray(fabio.open(path).data, dtype=np.float64)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    return img


def st(r, joint=False):
    if r is None:
        return "None"
    if joint:
        return (f"{r[0]:<8} m={r[1][3]:>2} rms={r[1][4]:6.2f} "
                f"ps={r[1][6]:+.3f} λ̂={r[2]:.4f}")
    return (f"{r[0]:<8} m={r[1][3]:>2} rms={r[1][4]:6.2f} "
            f"ps={r[1][6]:+.3f} D={r[1][2]*1.0:.0f}px")


def run(path, poni, tiers=("default", "large"), joint=False):
    img = load_img(os.path.join(ZOO, path))
    ai = pyFAI.load(os.path.join(ZOO, poni))
    lam = float(ai.wavelength) * 1e10 if ai.wavelength else None
    pxn = float(ai.detector.pixel1)
    out = {}
    for tier in tiers:
        ms = 520 if tier == "default" else int(np.ceil(max(img.shape) / 2.0))
        c, s, _, _ = to_canvas(img, max_side=ms)
        t0 = time.perf_counter()
        if joint:
            r = solve_auto_lambda(c, pxn * s, auto_center=True)
        else:
            r = solve_auto(c, lam, pxn * s, cascade=True, auto_center=True,
                           refine_2d=True)
        dt = time.perf_counter() - t0
        print(f"  [{tier} s={s} {c.shape[1]}x{c.shape[0]} {dt:6.1f}s] "
              f"{st(r, joint=joint)}", flush=True)
        out[tier] = r
    return out


print("== A. c-risk 必须拒收：Eiger4M / Eiger9M（s=5/7，大档单点 Δps 应 <0.05）", flush=True)
for p, po in [("Eiger4M.edf", "Eiger4M.poni"), ("Eiger9M.h5", "Eiger9M.poni")]:
    print(f"[{p}]", flush=True)
    o = run(p, po)
    if o["default"] and o["large"]:
        dps = o["large"][1][6] - o["default"][1][6]
        dm = o["large"][1][3] - o["default"][1][3]
        print(f"  → Δps={dps:+.3f} Δm={dm:+d}  "
              f"c 门应拒收: {not (dps >= 0.05 and dm >= 0)}", flush=True)

print("\n== B. mar3450：默认档（期望 Quartz 假解）+ 大档单点（b 还是 c 捕获？）", flush=True)
o = run("LaB6_260210.mar3450", "Mar3450.poni")

print("\n== C. 无 λ 联合：Frelon2k / halfccd（b 门不得误伤）", flush=True)
for p, po, ti in [("Frelon2k.edf", "Frelon2k.poni", ("default", "large")),
                  ("halfccd.edf", "halfccd.poni", ("default",))]:
    print(f"[{p}]", flush=True)
    run(p, po, tiers=ti, joint=True)

print("\n== D. 默认档统计（生产不扫：s<5 或锁表）", flush=True)
for p, po in [("Pilatus1M.edf", "Pilatus1M.poni"),
              ("Pilatus2M.cbf", "Pilatus2M.poni")]:
    print(f"[{p}]", flush=True)
    run(p, po, tiers=("default",))

print("\n== E. ceria 锁表模式（q8+rms1.8，默认档 rms 应 <1.8 → b 门不触发）", flush=True)
LAM_C, PXC = 0.972386, 172e-6
Q8 = phase_qs("CeO2")[:8]
for d in (150, 400):
    img = np.asarray(fabio.open(os.path.join(
        ROOT, "data", "real", "MX_ceria",
        f"ceria_{d}_1_0001.cbf")).data, dtype=np.float64)
    c1, s1, _, _ = to_canvas(img, max_side=520)
    r1 = solve_cascade(c1, LAM_C, PXC * s1, Q8, rms_floor=1.8)
    stt = "None" if r1 is None else f"m={r1[3]} rms={r1[4]:.2f} ps={r1[6]:+.3f}"
    print(f"  ceria_{d}: s={s1} {stt}", flush=True)
print("[done]", flush=True)
