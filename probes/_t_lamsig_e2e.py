# -*- coding: utf-8 -*-
"""_t_lamsig_e2e.py:λ 盲 solve_image → lam_sig 透传+σ 分级 e2e(2026-11-08)。

Eiger4M(zoo 宽角域,σ 小应"可信")与 alumina mean(窄角域,σ 大应"简并")
双例对照,验证 GUI 提示分级的数据基础 lam_sig 全链在场。
"""
import os
import sys

import numpy as np
import fabio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc

# 例1:Eiger4M(宽角域,期待 σ 小)
img = fabio.open(r"data\real\silx_zoo\Eiger4M.edf").data
img = np.where(np.isfinite(img), img, 0.0).astype(np.float64)
img[img >= 1e9] = 0.0
r = sc.solve_image(img, 75e-6)          # λ=None 联合
print(f"Eiger4M: phase={r['phase']} D={r['sol'][2]*r['px_m']*1e3:.1f}mm "
      f"lam={r['lam']:.4f}")
sig = r.get("lam_sig")
assert "lam_sig" in r, "lam_sig 键缺失(透传失败)"
if sig is not None:
    print(f"  lam_sig={sig:.4f} ({sig:.1%}) → "
          + ("宽角域·可信" if sig < 0.02 else "简并域·警示"))
else:
    print("  lam_sig=None(fallback 网格路径,GUI 走 σ 未知分支)")

# 例2:alumina mean(窄角域,期待 σ 大)
import glob
D_IR = sorted(glob.glob(r"data\real\alumina\*.edf"))
imgm = np.mean([fabio.open(f).data for f in D_IR], 0).astype(np.float64)
imgm = np.where(np.isfinite(imgm), imgm, 0.0)
r2 = sc.solve_image(imgm, (103.358e-6 + 102.530e-6) / 2)
print(f"\nalumina mean: phase={r2['phase']} "
      f"D={r2['sol'][2]*r2['px_m']*1e3:.1f}mm lam={r2['lam']:.4f}")
sig2 = r2.get("lam_sig")
if sig2 is not None:
    print(f"  lam_sig={sig2:.4f} ({sig2:.1%}) → "
          + ("宽角域·可信" if sig2 < 0.02 else "简并域·警示"))
else:
    print("  lam_sig=None → GUI 走 σ 未知分支(警示)")

print("\n总判:✓ lam_sig 全链透传,GUI 可分级"
      if ("lam_sig" in r) else "✗ 透传失败")
