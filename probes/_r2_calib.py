# -*- coding: utf-8 -*-
"""R2 门控校准探针（18-1104 需求）。

⚠ 本版作废：PS 重定向下 stdout 全缓冲，1500s 超时前一行未刷出即被杀。
有效数据见 _r2_calib2.py（无缓冲落盘 results/_r2_calib2.txt）与
_r2_gate_test.py（端到端，results/_r2_gate_test2.txt）。

对每个相关数据集实测：
  默认档 (max_side=520) 与 升级档 (ceil(maxdim/2)) 的 (phase, m, rms, ps)
  + 升级档 λ 网格扫描（复刻旧 _fallback_joint，仅 mar 有必要）

产出：两档成绩对照表 → 定 FB_RMS_GATE / FB_M_GATE / (c) 边际。
"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(os.path.join(ROOT, "data", "real", "silx_zoo"))

import fabio  # noqa: E402
import pyFAI.azimuthalIntegrator  # noqa: E402

from solver_core import solve_auto, solve_auto_lambda, to_canvas  # noqa: E402
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


def stat(r):
    if r is None:
        return "None"
    ph, sol = r
    return f"{ph:<8} m={sol[3]:>2} rms={sol[4]:6.2f} ps={sol[6]:+.3f} " \
           f"D={sol[2]*1:.1f}px"


def tiers(img):
    md = max(img.shape)
    c1, s1, _, _ = to_canvas(img, max_side=520)
    ms2 = int(np.ceil(md / 2.0))
    c2, s2, _, _ = to_canvas(img, max_side=ms2)
    return c1, s1, c2, s2


# ---------- zoo 有 λ 项：盲解两档 ----------
ZOO_L = [("Pilatus1M.edf", "Pilatus1M.poni"),
         ("Pilatus2M.cbf", "Pilatus2M.poni"),
         ("Eiger4M.edf", "Eiger4M.poni"),
         ("Eiger9M.h5", "Eiger9M.poni"),
         ("LaB6_260210.mar3450", "Mar3450.poni")]

print("=" * 100)
print("A. zoo 有 λ（盲解 solve_auto cascade+auto_center+refine_2d）")
for img_f, poni_f in ZOO_L:
    img = load_img(os.path.join(ZOO, img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    lam = ai.wavelength * 1e10
    pxn = float(ai.detector.pixel1)
    c1, s1, c2, s2 = tiers(img)
    t0 = time.perf_counter()
    r1 = solve_auto(c1, lam, pxn * s1, cascade=True, auto_center=True,
                    refine_2d=True)
    t1 = time.perf_counter() - t0
    print(f"\n[{img_f}] λ={lam:.4f} s_def={s1} s_lrg={s2} "
          f"({c1.shape[1]}x{c1.shape[0]} → {c2.shape[1]}x{c2.shape[0]})")
    print(f"  默认档 {t1:5.1f}s : {stat(r1)}")
    if s2 == s1:
        print("  升级档 = 默认档（s 相同，跳过）")
        continue
    if c2.shape[0] * c2.shape[1] > 3_500_000:
        print("  升级档面积 > 3.5Mpx（旧上限，跳过）")
        continue
    t0 = time.perf_counter()
    r2 = solve_auto(c2, lam, pxn * s2, cascade=True, auto_center=True,
                    refine_2d=True)
    t2 = time.perf_counter() - t0
    print(f"  升级档 {t2:5.1f}s : {stat(r2)}  (Δps={0 if r1 is None or r2 is None else r2[1][6]-r1[1][6]:+.3f})")
    if "mar" in img_f:
        # 复刻旧 _fallback_joint 网格（7 点）
        t0 = time.perf_counter()
        best = None
        for lg in np.geomspace(0.95, 1.05, 7) * lam:
            rr = solve_auto(c2, float(lg), pxn * s2, cascade=True,
                            auto_center=True, refine_2d=True)
            if rr is None:
                continue
            key = (rr[1][6], rr[1][3], -rr[1][4])
            if best is None or key > best[0]:
                best = (key, rr, float(lg))
        t3 = time.perf_counter() - t0
        print(f"  网格fb {t3:5.1f}s : {stat(best[1])} λ̂={best[2]:.4f}")

# ---------- zoo 无 λ 项（Frelon2k / halfccd）：联合 λ 两档 ----------
print("\n" + "=" * 100)
print("B. zoo 无 λ（solve_auto_lambda auto_center=True）")
for img_f, poni_f in [("Frelon2k.edf", "Frelon2k.poni"),
                      ("halfccd.edf", "halfccd.poni")]:
    img = load_img(os.path.join(ZOO, img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    pxn = float(ai.detector.pixel1)
    c1, s1, c2, s2 = tiers(img)
    t0 = time.perf_counter()
    r1 = solve_auto_lambda(c1, pxn * s1, auto_center=True)
    t1 = time.perf_counter() - t0
    st = ("None" if r1 is None else
          f"{r1[0]:<8} m={r1[1][3]:>2} rms={r1[1][4]:6.2f} "
          f"ps={r1[1][6]:+.3f} λ̂={r1[2]:.4f}")
    print(f"\n[{img_f}] s_def={s1} s_lrg={s2}")
    print(f"  默认档 {t1:5.1f}s : {st}")
    if s2 == s1:
        continue
    t0 = time.perf_counter()
    r2 = solve_auto_lambda(c2, pxn * s2, auto_center=True)
    t2 = time.perf_counter() - t0
    st2 = ("None" if r2 is None else
           f"{r2[0]:<8} m={r2[1][3]:>2} rms={r2[1][4]:6.2f} "
           f"ps={r2[1][6]:+.3f} λ̂={r2[2]:.4f}")
    d = (0 if r1 is None or r2 is None else r2[1][6] - r1[1][6])
    print(f"  升级档 {t2:5.1f}s : {st2}  (Δps={d:+.3f})")

# ---------- ceria 7 档（锁相 q8, rms_floor=1.8, 无 auto_center） ----------
print("\n" + "=" * 100)
print("C. ceria（phase_fit 口径：solve_cascade(Q8, rms_floor=1.8)）")
LAM_C = 0.972386
PXC = 172e-6
Q8 = phase_qs("CeO2")[:8]
from solver_core import solve_cascade  # noqa: E402
for d in (150, 200, 250, 300, 350, 400, 450):
    img = np.asarray(fabio.open(os.path.join(
        ROOT, "data", "real", "MX_ceria",
        f"ceria_{d}_1_0001.cbf")).data, dtype=np.float64)
    c1, s1, c2, s2 = tiers(img)
    t0 = time.perf_counter()
    r1 = solve_cascade(c1, LAM_C, PXC * s1, Q8, rms_floor=1.8)
    t1 = time.perf_counter() - t0
    st = ("None" if r1 is None else
          f"m={r1[3]:>2} rms={r1[4]:6.2f} ps={r1[6]:+.3f}")
    print(f"ceria_{d}: s_def={s1} s_lrg={s2}  默认 {t1:4.1f}s {st}")
    t0 = time.perf_counter()
    r2 = solve_cascade(c2, LAM_C, PXC * s2, Q8, rms_floor=1.8)
    t2 = time.perf_counter() - t0
    st2 = ("None" if r2 is None else
           f"m={r2[3]:>2} rms={r2[4]:6.2f} ps={r2[6]:+.3f}")
    d = (0 if r1 is None or r2 is None else r2[6] - r1[6])
    print(f"               升级 {t2:4.1f}s {st2}  (Δps={d:+.3f})")
print("\n[done]")
