# -*- coding: utf-8 -*-
r"""_id22_duel.py — ID22 2θ 扫描 9 帧 V3 零样本族检验（Eiger2 CdTe 2M-W, LaB6 @ 35keV）。

数据：eiger_0000.h5（bitshuffle/LZ4，需 hdf5plugin；focus.h5 内嵌 data 全 0 不可用）
硬真值：tth 编码器增量 ≈ +10.000°/帧（nom 0→80），探测器绕样品 2θ 臂旋转。
无官方 poni/npt → 本检验 = V3 帧间 SO(3) 增量 vs 电机真值 + 相鉴（期望 LaB6）+ D/中心随 tth 物理趋势。
"""
import sys

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from PyQt5.QtWidgets import QApplication
app = QApplication([])
import i18n
i18n.set_lang("zh")

import hdf5plugin  # noqa: F401  (注册 bitshuffle 解码器, 必须在 h5py 读前)
import h5py

from gui_solver import SolveWorker, hybrid_ai

BASE = r"C:\Users\liu\Desktop\衍射测试图\ext1\gonio_ID22"
LAM_A = 0.3542403306673354   # positioners/wavelength
PX_UM = 75.0

with h5py.File(BASE + r"\eiger_0000.h5", "r") as f:
    FRAMES = f["entry_0000/instrument/eiger/data"][()].astype(np.float64)
    TTH = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
print("frames:", FRAMES.shape, flush=True)


def rmat(r1, r2, r3):
    sx, cx = np.sin(r1), np.cos(r1)
    sy, cy = np.sin(r2), np.cos(r2)
    sz, cz = np.sin(r3), np.cos(r3)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def angle_axis(R):
    tr = np.clip(np.trace(R), -1.0, 3.0)
    th = np.degrees(np.arccos((tr - 1) / 2))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(v)
    return th, (v / n if n > 1e-12 else np.array([np.nan] * 3))


rows = []
for i in range(FRAMES.shape[0]):
    img = FRAMES[i]
    try:
        box = {}
        w = SolveWorker(img, LAM_A, PX_UM)
        w.done.connect(lambda res: box.update(r=res))
        w.run()
        res = box.get("r")
    except Exception as exc:
        rows.append(dict(tag="tth%02d" % TTH[i], ph="ERR", err=repr(exc)))
        print("tth%02.0f: SOLVE ERROR %r" % (TTH[i], exc), flush=True)
        continue
    if not res:
        rows.append(dict(tag="tth%02d" % TTH[i], ph="FAIL", err="no result"))
        print("tth%02.0f: V3 SOLVE FAIL" % TTH[i], flush=True)
        continue
    ai = hybrid_ai(res, PX_UM)
    f2 = ai.getFit2D()
    rows.append(dict(tag="tth%02d" % TTH[i], ph=res.get("ph"),
                     D=ai.dist * 1000, c=(f2["centerX"], f2["centerY"]),
                     rot=(ai.rot1, ai.rot2, ai.rot3), ai=ai))
    print("tth%02.0f: ph=%-7s D=%8.1fmm c=(%8.1f,%8.1f) rot=(%+.4f,%+.4f,%+.4f)"
          % (TTH[i], res.get("ph"), ai.dist * 1000, f2["centerX"], f2["centerY"],
             np.degrees(ai.rot1), np.degrees(ai.rot2), np.degrees(ai.rot3)), flush=True)

# -- 帧间 SO(3) 增量 vs 电机真值 --
ok = [r for r in rows if "ai" in r]
print("\n== SO(3) increments vs motor truth (enc step 10.000deg) ==", flush=True)
inc_lines = []
R = {r["tag"]: rmat(*[np.radians(x) for x in r["rot"]]) for r in ok}
for k in range(1, len(ok)):
    a, b = ok[k - 1]["tag"], ok[k]["tag"]
    th, ax = angle_axis(R[b] @ R[a].T)
    line = "%s->%s: V3 %8.3f deg (truth 10.000) axis=%s" % (a, b, th, np.round(ax, 3))
    inc_lines.append(line)
    print(line, flush=True)
if len(ok) >= 2:
    base = ok[0]["tag"]
    line = "cumulative %s->%s: V3 %8.3f deg (truth %d) " % (
        base, ok[-1]["tag"], angle_axis(R[ok[-1]["tag"]] @ R[base].T)[0],
        10 * (len(ok) - 1))
    inc_lines.append(line)
    print(line, flush=True)

# -- 相鉴统计 --
phs = [r.get("ph") for r in rows]
print("\nphase votes:", {p: phs.count(p) for p in set(phs) if p}, flush=True)

# -- md --
md = ["# ID22 2θ 扫描 9 帧 V3 零样本族检验（Eiger2 CdTe 2M-W, LaB6@0.35424Å, tth 0→80 步10°）", "",
      "数据 eiger_0000.h5（bitshuffle 需 hdf5plugin；focus.h5 内嵌 data 全 0 弃用）。",
      "无官方 poni/npt；硬真值 = tth 编码器帧间增量 ≈ +10.000°（2θ 臂绕样品转）。", "",
      "| tth(°) | V3 相 | D(mm) | center(px) | rot1/2/3(°) |", "|---|---|---|---|---|"]
for r in rows:
    if "ai" in r:
        md.append("| %s | %s | %.1f | (%.1f,%.1f) | %+.2f/%+.2f/%+.2f |" % (
            r["tag"][3:], r["ph"], r["D"], r["c"][0], r["c"][1],
            *np.degrees(r["rot"])))
    else:
        md.append("| %s | %s | - | - | - |" % (r["tag"][3:], r["ph"]))
md += ["", "## 帧间 SO(3) 增量 vs 电机 10° 真值", ""] + ["- " + l for l in inc_lines]
md += ["", "相鉴票型: %s" % {p: phs.count(p) for p in set(phs) if p}, ""]
with open(ROOT + r"\docs\nc_id22.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")
print("\nmd -> docs/nc_id22.md", flush=True)
