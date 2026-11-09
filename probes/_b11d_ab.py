# -*- coding: utf-8 -*-
"""_b11d_ab.py:B11d 分裂高斯定心 A/B(2026-11-07,真实数据判决)。

B11c 合成判决:分裂高斯把 0.3px 径向偏斜残余 0.179→0.026px(11×)。
本探针:生产径向谱定心 抛物线 vs 抛物线+splitG 精化(_SPLIT_G 开关),
ceria 七档真实帧 + P2M,比 |ΔD|/rms/m。判决:
  ① 无档劣化(ΔD 增幅 >0.05mm 或 m 掉环)→ 回退 _SPLIT_G=False;
  ② 任一档 ΔD 改善 ≥0.05mm 或 rms 全面降 → 保留并记录。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)

import solver_core as sc
from gui_solver import load_any

LAM = 0.972386
LAM_P2M = 12.398 / 72.1


def run(tag, path, gt_mm, px_um=172.0, lam=LAM):
    img = load_any(path)
    out = {}
    for label, flag in (("splitG", True), ("parab", False)):
        sc._SPLIT_G = flag
        sc.clear_masks()
        r = sc.solve_image(img, px_um * 1e-6, lam=lam)
        if r is None:
            out[label] = None
            continue
        out[label] = dict(ph=r["phase"],
                          D=r["sol"][2] * r["s"] * px_um * 1e-3,
                          ps=r["ps"], m=r["m"], rms=r["rms"])
    sc._SPLIT_G = True
    print(f"\n== {tag} GT={gt_mm}mm ==")
    for label in ("parab", "splitG"):
        d = out[label]
        if d is None:
            print(f"  {label:7} 解不出")
            continue
        print(f"  {label:7} ph={d['ph']:6} D={d['D']:8.2f} "
              f"Δ={abs(d['D'] - gt_mm):6.3f}  ps={d['ps']:.3f} m={d['m']:2} "
              f"rms={d['rms']:.2f}")
    return out


rows = []
GT = {150: 149.045, 200: 199.173, 250: 249.246, 300: 299.322,
      350: 349.383, 400: 399.459, 450: 449.48}   # 2026-11-07 读 poni 修正
for d_mm in (150, 200, 250, 300, 350, 400, 450):
    rows.append(run(f"ceria_{d_mm}",
                    ROOT + rf"\data\real\MX_ceria\ceria_{d_mm}_1_0001.cbf",
                    GT[d_mm]))
rows.append(run("P2M CdTe 800mm",
                ROOT + r"\data\real\Pilatus2MCdTe_ID15_CeO2_72100eV_800mm_0000.cbf",
                800.0, lam=LAM_P2M))

print("\n== 判决 ==")
worse = better = 0
for (tag, _) in zip(("150", "200", "250", "300", "350", "400", "450", "P2M"),
                    rows):
    pass
for tag, out in zip(("150", "200", "250", "300", "350", "400", "450", "P2M"),
                    rows):
    a, b = out["parab"], out["splitG"]
    if a is None or b is None:
        print(f"  {tag}: 有档解不出 → 查")
        worse += 1
        continue
    dd = abs(b["D"] - a["D"])
    imp = (abs(a["D"] - 200.19) if False else 0)
    if b["m"] < a["m"] or b["ps"] < a["ps"] - 0.02:
        print(f"  {tag}: m/ps 劣化 → 疑")
        worse += 1
    else:
        print(f"  {tag}: Δ(新-旧)={b['D'] - a['D']:+.3f}mm rms "
              f"{a['rms']:.2f}→{b['rms']:.2f}px m {a['m']}→{b['m']}")
print(f"\n结论:{'全部通过,保留 _SPLIT_G=True' if worse == 0 else f'{worse} 档疑劣化'}")
