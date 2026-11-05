# -*- coding: utf-8 -*-
"""_t_spl_reg.py：验收 B——无 spline 全链路回归（向导真实口径，基线金丝雀）。

历史（2026-11-03）：本脚本初版断言"check_std 判语必须含 ✓"，出生即红——
向导口径（SolveWorker+check_std）在 ceria_400（T7 拒收、5 环部分弧）
判语"可靠环不足"、ceria_150（T7 接收 8 环）判语"⚠谨慎 0.166°"，均非
✓ 档。经 _t_spl_ab.py 对改动前快照（_gui_solver_pre.py）A/B 对照，
12/12 字段比特级一致——该质量状态是**既有**的，与 T-SPL 接线无关
（F6：不选 spline 行为逐项一致）。

因此本脚本的职责改为**锁基线**：同图同参复跑，逐字段对齐 2026-11-03
基线值（容差紧）。任何未来改动（中心真值并入、精化策略调整……）若
改变了向导口径的数字，这里会立刻报警并指出是哪个字段。
判语本身也锁——它记录的是当前质量水位，后续提升应连基线一起更新。

两臂：
  ceria_400（桌面副本）——T7 拒收路径（画布 D + fit_tilt poni/rot）
  ceria_150（生产数据）——T7 接收路径（E1f2 起含原生倾角：V3 成对几何）

2026-11-03 深夜 E1f2 重锁：ceria_150 tilt_on False→True、自检 n 8→9、
判词"⚠谨慎 0.166°"→"0.045° ✓ 合格"（原生图倾角拟合，_nc_e1f2_ab.py
A/B 判决 V3：Σ锐度 10×、中心 6.65→1.24px、ΔD 持平；D/cx/cy 等
15 位一致）。
2026-11-03 深夜 E1f3b 重锁：V3 放开到非 native tilt_on 帧——ceria_400
自检 n 1→5、判词"可靠环不足"→"0.024° ✓ 合格"（_nc_e1f3b.py：dev
0.095→0.024°、Σ 17×、中心 24.5→2.5px、ΔD +0.11 持平）。
"""
import sys

sys.path.insert(0, r"D:\pyfai-project\probes")
sys.path.insert(0, r"D:\pyfai-project")
from PyQt5.QtWidgets import QApplication  # noqa
app = QApplication([])
import i18n
i18n.set_lang("zh")

from gui_solver import load_any, sniff_meta, SolveWorker, check_std

# 2026-11-03 基线（T-SPL 接线当日，A/B 对照逐字段一致；
# E1f2 深夜重锁 ceria_150 倾角臂）
BASE = {
    r"C:\Users\liu\Desktop\衍射测试图\ceria_400_1_0001.cbf": dict(
        ph="CeO2", D_mm=399.5687, cx=1249.61, cy=1274.62,
        native=False, tilt_on=True, center_src="solver",
        ring_rms=0.3765, n=5, verdict_key="0.024° ✓ 合格"),
    r"D:\pyfai-project\data\real\MX_ceria\ceria_150_1_0001.cbf": dict(
        ph="CeO2", D_mm=148.97, cx=1227.5, cy=1257.8,
        native=True, tilt_on=True, center_src="r2d",
        ring_rms=None, n=9, verdict_key="0.045° ✓ 合格"),
}

fails = []
for img_path, base in BASE.items():
    img = load_any(img_path)
    meta = sniff_meta(img_path)
    lam = float("{:g}".format(meta["lam"])) if "lam" in meta else None
    box = {}
    w = SolveWorker(img, lam, 172.0)
    w.done.connect(lambda r: box.update(r=r))
    w.fail.connect(lambda tb: (_ for _ in ()).throw(RuntimeError(tb)))
    w.run()
    res = box["r"]
    assert res, "解不出"
    c = check_std(img, res, 172.0)
    got = dict(ph=res["ph"], D_mm=res["D_px"] * 172e-3,
               cx=res["cx"], cy=res["cy"], native=res["native"],
               tilt_on=res["tilt_on"], center_src=res["center_src"],
               ring_rms=res["ring_rms"], n=c["n"], verdict_key=c["verdict"])
    print(f"\n—— {img_path.split(chr(92))[-1]} ——")
    for k, b in base.items():
        g = got[k]
        if isinstance(b, float) and isinstance(g, float):
            # 画布确定性求解应精确复现；±0.06 只挡跨平台浮点噪声
            ok = abs(g - b) < 0.06
        elif k == "verdict_key":
            ok = isinstance(g, str) and b in g
        elif b is None:
            ok = True
        else:
            ok = g == b
        mark = "✓" if ok else "✗"
        print(f"  {k:<12} 基线={str(b):<22} 实测={str(g):<22} {mark}")
        if not ok:
            fails.append((img_path, k, b, g))

print()
if fails:
    for f in fails:
        print(f"✗ 回归：{f[0]} 字段 {f[1]} 基线 {f[2]} → 实测 {f[3]}")
    print("❌ 向导口径基线被打破——若是有意改动请同步更新本表基线")
    sys.exit(1)
print("✅ 验收 B 通过：无 spline 时向导口径与 2026-11-03 基线逐字段一致")
print("   （E1f3b 起水位：ceria_400 0.024° ✓ 合格、ceria_150 0.045° ✓ 合格）")
