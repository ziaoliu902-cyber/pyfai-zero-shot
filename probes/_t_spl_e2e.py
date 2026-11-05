# -*- coding: utf-8 -*-
"""_t_spl_e2e.py：T-SPL 端到端验收 A（合成畸变往返，2026-11-03 需求 §5A）。

流程：
  clean(ceria_150 实测图) --frelon.spline 正向揉歪--> raw_sim
  三臂对比：
    B0  clean 直解            → 基准 (D0, c0)
    B1  raw_sim 直解（不校正） → 应出现可测偏差（ΔD>1mm 或 中心>5px）
    B2  raw_sim 校正后解      → 必须回到基值（|ΔD|<0.1mm、中心<1px）
  另验 F4：导出 poni 带 SplineFile，pyFAI.load 回读一致。

用法：python -X utf8 probes/_t_spl_e2e.py
"""
import os
import sys
import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from PyQt5.QtWidgets import QApplication  # noqa：SolveWorker 是 QThread
app = QApplication([])

from scipy.ndimage import map_coordinates
import pyFAI
from pyFAI.spline import Spline

from gui_solver import (load_any, sniff_meta, SolveWorker, apply_spline,
                        hybrid_ai, set_splinefile)

IMG = os.path.join(ROOT, r"data\real\MX_ceria\ceria_150_1_0001.cbf")
PONI = os.path.join(ROOT, r"data\real\MX_ceria\ceria_150_1_0001.poni")
SPL = os.path.join(ROOT, r"data\real\silx_zoo\frelon.spline")


def solve_sync(img, lam, px_um):
    """同步跑 SolveWorker（直调 run()，不等事件循环）。"""
    box = {}
    w = SolveWorker(img, lam, px_um)
    w.done.connect(lambda r: box.update(r=r))
    w.fail.connect(lambda tb: (_ for _ in ()).throw(RuntimeError("solve fail:\n" + tb)))
    w.run()          # 同步执行（绕过线程）
    return box["r"]


def main():
    clean = load_any(IMG)
    meta = sniff_meta(IMG)
    px = float(meta.get("px", 172))
    lam_A = pyFAI.load(PONI).wavelength * 1e10
    print(f"图 {clean.shape}  px={px:g}µm  λ={lam_A:.4f}Å")

    # ── 正向揉歪（_t5_roundtrip 同款：raw=map(clean,[yy+DY,xx+DX])）──
    sp = Spline(SPL)
    H, W = clean.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    DX = np.asarray(sp.splineFuncX(xx, yy), float)
    DY = np.asarray(sp.splineFuncY(xx, yy), float)
    print(f"frelon.spline 幅度：max|DX|={np.abs(DX).max():.1f}px  max|DY|={np.abs(DY).max():.1f}px")
    raw_sim = map_coordinates(clean, [yy + DY, xx + DX], order=1,
                              mode="nearest")

    # ── 三臂解算 ──
    print("\n[B0] clean 直解 …", flush=True)
    r0 = solve_sync(clean, lam_A, px)
    assert r0 is not None, "B0 解不出（图/参数有问题）"
    print(f"  D={r0['D_px']*px*1e-3:.2f}mm  c=({r0['cx']:.1f},{r0['cy']:.1f})")

    print("[B1] raw_sim 不校正直解 …", flush=True)
    r1 = solve_sync(raw_sim, lam_A, px)
    if r1 is None:
        print("  解不出——也算'可测劣化'（信息被畸变破坏）")
        b1_bad = True
    else:
        dD1 = (r1["D_px"] - r0["D_px"]) * px * 1e-3
        dc1 = max(abs(r1["cx"] - r0["cx"]), abs(r1["cy"] - r0["cy"]))
        print(f"  ΔD={dD1:+.2f}mm  Δ中心=({r1['cx']-r0['cx']:+.1f},{r1['cy']-r0['cy']:+.1f})px")
        b1_bad = (abs(dD1) > 1.0) or (dc1 > 5.0)

    print("[B2] raw_sim 校正后解 …", flush=True)
    corr = apply_spline(raw_sim, SPL)
    r2 = solve_sync(corr, lam_A, px)
    assert r2 is not None, "B2 解不出——校正图求解失败！"
    dD2 = (r2["D_px"] - r0["D_px"]) * px * 1e-3
    dc2 = max(abs(r2["cx"] - r0["cx"]), abs(r2["cy"] - r0["cy"]))
    print(f"  ΔD={dD2:+.3f}mm  Δ中心=({r2['cx']-r0['cx']:+.2f},{r2['cy']-r0['cy']:+.2f})px")

    # ── F4：poni SplineFile 回读 ──
    ai = hybrid_ai(r2, px)
    set_splinefile(ai, SPL)
    tmp = os.path.join(os.environ["TEMP"], "_spl_e2e.poni")
    ai.save(tmp)
    d = pyFAI.load(tmp).detector
    back = getattr(d, "splinefile", None) or getattr(d, "splineFile", None)
    ok_f4 = bool(back) and os.path.normcase(back) == os.path.normcase(SPL)
    print(f"\n[F4] poni 回读 splineFile: {back}  {'✓' if ok_f4 else '✗'}")

    # ── 判决 ──
    print("\n===== 验收 A 判决 =====")
    print(f"B1 不校正应劣化 : {'✓' if b1_bad else '✗（畸变幅度可能太小，换样条）'}")
    print(f"B2 |ΔD|<0.1mm   : {'✓' if abs(dD2) < 0.1 else '✗'} ({dD2:+.3f}mm)")
    print(f"B2 中心<1px     : {'✓' if dc2 < 1.0 else '✗'} ({dc2:.2f}px)")
    print(f"F4 SplineFile   : {'✓' if ok_f4 else '✗'}")
    ok = (b1_bad or r1 is None) and abs(dD2) < 0.1 and dc2 < 1.0 and ok_f4
    print("总体：" + ("✅ 通过" if ok else "❌ 不通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
