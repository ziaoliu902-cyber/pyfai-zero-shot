# -*- coding: utf-8 -*-
"""R2 门控端到端测试：solve_image 在 zoo 关键项上的判决与耗时。

预期：
  P1M/P2M     s=3/4 → 无扫描，行不变
  Eiger4M     c-risk 探针 → D 分叉 ~0.2% → 拒收（行不变）
  Eiger9M     同上（~0.07%）
  mar3450     c-risk 探针 → 分叉 ≥10% → λ 网格救援 → LaB6（旧 +5.87 路径）
  Frelon2k/halfccd 联合 → s=4 → 无扫描（b 门 m=3 不触发）
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
import pyFAI.azimuthalIntegrator  # noqa: E402

from solver_core import solve_image  # noqa: E402

ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
os.chdir(ZOO)          # Frelon2k poni 引用相对路径 frelon.spline


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


for img_f, poni_f in [("Pilatus1M.edf", "Pilatus1M.poni"),
                      ("Pilatus2M.cbf", "Pilatus2M.poni"),
                      ("Eiger4M.edf", "Eiger4M.poni"),
                      ("Eiger9M.h5", "Eiger9M.poni"),
                      ("LaB6_260210.mar3450", "Mar3450.poni"),
                      ("Frelon2k.edf", "Frelon2k.poni"),
                      ("halfccd.edf", "halfccd.poni")]:
    img = load_img(os.path.join(ZOO, img_f))
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator.sload(poni_f)
    lam = float(ai.wavelength) * 1e10 if ai.wavelength else None
    pxn = float(ai.detector.pixel1)
    t0 = time.perf_counter()
    res = solve_image(img, pxn, lam=lam)
    dt = time.perf_counter() - t0
    if res is None:
        print(f"{img_f:<24} → None（两档均失败）  {dt:6.1f}s", flush=True)
        continue
    D_mm = res["sol"][2] * res["px_m"] * 1000
    print(f"{img_f:<24} {str(res['phase']):<8} m={res['m']:>2} "
          f"rms={res['rms']:5.2f} ps={res['ps']:+.3f} D={D_mm:8.1f}mm "
          f"s={res['s']} fb={res['used_fallback']} reason={res['fb_reason']} "
          f"λ̂={res['lam']:.4f}  {dt:6.1f}s  "
          f"(def: m={res['def_m']} rms={res['def_rms']:.2f} "
          f"ps={res['def_ps']:+.3f} | lrg: m={res['lrg_m']} "
          f"rms={res['lrg_rms']} ps={res['lrg_ps']})", flush=True)
print("[done]", flush=True)
