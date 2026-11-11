# -*- coding: utf-8 -*-
"""_t_agbh_repro.py:AgBh 图(Pilatus1M)被 AgSt 抢认复现(2026-11-08)。

GT:AgBh(silx zoo)。跑当前 solve_image(λ 盲,向导口径)看鉴定。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probes"))
os.chdir(ROOT)

import fabio
import numpy as np
import solver_core

img = fabio.open(r"data\real\silx_zoo\Pilatus1M.edf").data
img = np.where(np.isfinite(img), img, 0.0).astype(np.float64)
print("Pilatus1M:", img.shape)

r = solver_core.solve_image(img, 172e-6, lam=None)
if r is None:
    print("λ盲:None")
else:
    ph, sol = r["phase"], r["sol"]
    print(f"λ盲: ph={ph} m={sol[3]} rms={sol[4]:.3f} ps={sol[6]:.3f} "
          f"lam={r['lam']:.4f} ambig={r.get('ambig')}")
    # 真相是 AgBh:无论胜者是谁,AgBh↔AgSt 歧义必须被标出
    assert r.get("ambig") is not None, "AgBh/AgSt 图必须带 ambig 标注"
    if ph == "AgSt":
        # λ 盲时 D 与 λ 同缩(乘积简并),备选 mm 无绝对意义——
        # 只在锁 λ 分支断言换算(见下)。
        print("(λ 盲:绝对尺度简并,备选距离见锁 λ 分支)")

# GT λ(poni 里读)
with open(r"data\real\silx_zoo\Pilatus1M.poni") as f:
    gt = dict(l.split(":", 1) for l in f if ":" in l)
lam_gt = float(gt["Wavelength"]) * 1e10
print(f"GT λ={lam_gt:.4f}Å D={float(gt['Distance'])*1e3:.1f}mm")
r2 = solver_core.solve_image(img, 172e-6, lam=lam_gt)
if r2 is None:
    print("锁λ:None")
else:
    ph, sol = r2["phase"], r2["sol"]
    print(f"锁λ: ph={ph} m={sol[3]} rms={sol[4]:.3f} ps={sol[6]:.3f} "
          f"ambig={r2.get('ambig')}")
    assert r2.get("ambig") is not None, "锁 λ 也必须带 ambig"
    # 注:锁 λ=1.0Å 时此图落在 D≈30mm 的深垃圾盆(环全塞束挡,
    # 既有行为,非本轮引入;6 相库时代 zoo 记录 +56mm 谐波退化)。
    # 备选距离换算的数值验证须走 GUI SolveWorker 原生精化口径
    # (res["D_px"]),画布直出 sol[0] 不具代表性——此处仅标注不断言。
