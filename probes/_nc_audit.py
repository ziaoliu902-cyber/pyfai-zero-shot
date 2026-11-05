# -*- coding: utf-8 -*-
r"""_nc_audit.py — NC 核心证据 v0：公开标定文件的共心真值审计。

对本地每一个 (image, .poni) 对：
  ① 零样本求解（SolveWorker 生产路径，GT 只在评估侧）
  ② concentric_center 出共心真值 ref（σ 曲率估计）
  ③ 记录：我们Δc(对ref) ｜ GTΔc(poni中心对ref) ｜ σ_ref ｜ ΔD ｜ 相鉴定
输出：docs/nc_audit_v0.md + .json
协议：λ 取 poni 名义值（zoo 基准同口径；无 λ 模式另见 zoo_benchmark_v3）；
px 取 poni 探测器规格（仪器参数，非标定量）。共心环 <2 → ref 不可用，如实记。
"""
import json
import os
import sys
import time

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from PyQt5.QtWidgets import QApplication
app = QApplication([])
import i18n
i18n.set_lang("zh")

import fabio
import pyFAI

from gui_solver import load_any, SolveWorker
from solver_core import concentric_center
from torchfai.phases import phase_qs

ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
CERIA = os.path.join(ROOT, "data", "real", "MX_ceria")
ALUMINA = os.path.join(ROOT, "data", "real", "alumina")

PAIRS = []                                    # (tag, img, poni)
for d in (150, 200, 250, 300, 350, 400, 450):  # ceria：7 档
    PAIRS.append((f"ceria_{d}",
                  os.path.join(CERIA, f"ceria_{d}_1_0001.cbf"),
                  os.path.join(CERIA, f"ceria_{d}_1_0001.poni")))
for img_f, poni_f, tag in [
        ("Pilatus1M.edf", "Pilatus1M.poni", "Pilatus1M"),
        ("Pilatus2M.cbf", "Pilatus2M.poni", "Pilatus2M"),
        ("Pilatus6M.cbf", "Pilatus6M.poni", "Pilatus6M"),
        ("Eiger4M.edf", "Eiger4M.poni", "Eiger4M"),
        ("Eiger4M.edf", "Eiger4M_v2.poni", "Eiger4M_v2"),
        ("Eiger9M.h5", "Eiger9M.poni", "Eiger9M"),
        ("Frelon2k.edf", "Frelon2k.poni", "Frelon2k"),
        ("halfccd.edf", "halfccd.poni", "halfccd"),
        ("halfccd.edf", "LaB6.poni", "LaB6_0020poni"),   # 同图不同 poni 对照
        ("LaB6_260210.mar3450", "Mar3450.poni", "mar3450"),
        ("Fairchild.edf", "Fairchild.poni", "Fairchild")]:
    PAIRS.append((tag, os.path.join(ZOO, img_f), os.path.join(ZOO, poni_f)))
_poni_al = [f for f in os.listdir(ALUMINA) if f.endswith(".poni")]
if _poni_al:
    _p = os.path.join(ALUMINA, _poni_al[0])
    for f in sorted(os.listdir(ALUMINA)):
        if f.endswith(".edf"):
            PAIRS.append((f"alumina/{f[:11]}", os.path.join(ALUMINA, f), _p))

print(f"审计对象：{len(PAIRS)} 个 (image, poni) 对", flush=True)
rows, jout = [], []
t_all = time.perf_counter()
for tag, img_f, poni_f in PAIRS:
    try:
        ai = pyFAI.load(poni_f)
        try:
            f2d = ai.getFit2D()
            gx, gy = f2d["centerX"], f2d["centerY"]
        except Exception:
            gx = ai.poni2 / ai.detector.pixel2
            gy = ai.poni1 / ai.detector.pixel1
        px_um = float(ai.detector.pixel1) * 1e6
        lam = (ai.wavelength * 1e10) if ai.wavelength else 0.972386
        gt_D = ai.dist * 1000.0

        img = load_any(img_f)
        img = np.where(img >= 4e9, 0.0, img)   # E9M uint32 哨兵（T7 教训）
        box = {}
        w = SolveWorker(img, lam, px_um)
        w.done.connect(lambda r: box.update(r=r))
        w.fail.connect(lambda tb: box.update(r=None, tb=tb))
        t0 = time.perf_counter()
        w.run()
        ms = (time.perf_counter() - t0) * 1000
        res = box.get("r")
        if res is None:
            rows.append(f"| {tag} | SOLVE FAIL | — | — | — | — | — |")
            jout.append(dict(tag=tag, ok=False, reason="solve-fail"))
            print(f"{tag:22} SOLVE FAIL", flush=True)
            continue

        ph, cx, cy, D_px = res["ph"], res["cx"], res["cy"], res["D_px"]
        D_mm = D_px * px_um * 1e-3
        cr = concentric_center(img, (cx, cy), D_px,
                               phase_qs(ph), lam)
        if cr is None:
            ours_dc = gt_dc = sig = n = None
            cells = (f"| {ph}✔ | — | {np.hypot(cx-gx, cy-gy):.1f}ᴳᵀ | "
                     f"{D_mm-gt_D:+.2f} | ref不可用(环<2) |")
        else:
            ours_dc = float(np.hypot(cx - cr["cx"], cy - cr["cy"]))
            gt_dc = float(np.hypot(gx - cr["cx"], gy - cr["cy"]))
            sig = float(max(cr["sx"], cr["sy"]))
            n = int(cr["n"])
            cells = (f"| {ph}✔ | ({cr['cx']:.1f},{cr['cy']:.1f}) | "
                     f"{n} | {sig:.3f} | **{ours_dc:.2f}** | "
                     f"**{gt_dc:.1f}** | {D_mm-gt_D:+.2f} |")
        rows.append(f"| {tag} | " + cells[1:])
        jout.append(dict(tag=tag, ok=True, phase=str(ph), n_ref=n,
                         cref=[cr["cx"], cr["cy"]] if cr else None,
                         sigma_ref=sig, ours_dc=ours_dc, gt_dc=gt_dc,
                         dD=float(D_mm - gt_D), ms=ms,
                         gt_center=[float(gx), float(gy)]))
        _o = f"{ours_dc:.2f}" if ours_dc is not None else "—"
        _g = f"{gt_dc:.1f}" if gt_dc is not None else "—"
        print(f"{tag:22} {ph:8} 我们Δc={_o:>6}  GTΔc={_g:>6}"
              f"  σ={sig:.3f}" if sig else
              f"{tag:22} {ph:8} ref不可用", flush=True)
    except Exception as e:
        rows.append(f"| {tag} | ERROR {type(e).__name__}: {e} |")
        jout.append(dict(tag=tag, ok=False, reason=f"{type(e).__name__}: {e}"))
        print(f"{tag:22} ERROR {e}", flush=True)

# ---- 汇总统计 ----
gt_all = [j["gt_dc"] for j in jout if j.get("gt_dc") is not None]
our_all = [j["ours_dc"] for j in jout if j.get("ours_dc") is not None]
per_file = {}
for j in jout:                                # 每标定文件取首帧 GTΔc（alumina 去重）
    if j.get("gt_dc") is not None and j["tag"] not in per_file:
        per_file[j["tag"].split("/")[0]] = j["gt_dc"]
gf = np.array(sorted(per_file.values()))
oa = np.array(our_all)
summ = dict(
    n_pairs=len(PAIRS), n_ok=sum(1 for j in jout if j.get("ok")),
    n_ref=len(our_all),
    gt_audit=dict(n_files=len(gf), median=float(np.median(gf)),
                  p90=float(np.percentile(gf, 90)), max=float(gf.max()),
                  frac_gt5px=float((gf > 5).mean()),
                  frac_gt1px=float((gf > 1).mean())),
    ours=dict(n=len(oa), median=float(np.median(oa)),
              p90=float(np.percentile(oa, 90)), max=float(oa.max())),
)
print("\n==== 汇总 ====")
print(f"GT 审计（{summ['gt_audit']['n_files']} 个标定文件）："
      f"median {summ['gt_audit']['median']:.1f}px  "
      f"p90 {summ['gt_audit']['p90']:.1f}px  "
      f"max {summ['gt_audit']['max']:.1f}px  "
      f">5px 占比 {summ['gt_audit']['frac_gt5px']*100:.0f}%")
print(f"我们精度（{summ['ours']['n']} 帧）："
      f"median {summ['ours']['median']:.2f}px  "
      f"p90 {summ['ours']['p90']:.2f}px  max {summ['ours']['max']:.2f}px")

md = ["# NC 核心证据 v0：公开标定文件的共心真值审计",
      "",
      f"生成：_nc_audit.py ｜ {len(PAIRS)} 对 (image, poni) ｜ "
      f"协议：零样本求解（GT 仅评估侧）→ concentric_center 共心真值 → "
      f"双口径 Δc。λ=poni 名义值；px=poni 探测器规格。",
      "",
      "| 标定文件 | 相(我们) | center_ref | 环数 | σ_ref(px) | "
      "我们Δc(px) | **GTΔc(px)** | ΔD(mm) |",
      "|---|---|---|---|---|---|---|---|"] + rows + [
      "",
      f"**GT 审计**：N={summ['gt_audit']['n_files']} 文件，"
      f"median **{summ['gt_audit']['median']:.1f}px**、"
      f"p90 {summ['gt_audit']['p90']:.1f}px、"
      f"max **{summ['gt_audit']['max']:.1f}px**、"
      f">5px 占比 **{summ['gt_audit']['frac_gt5px']*100:.0f}%**。",
      f"**我们**：N={summ['ours']['n']} 帧，"
      f"median **{summ['ours']['median']:.2f}px**、"
      f"p90 {summ['ours']['p90']:.2f}px、max {summ['ours']['max']:.2f}px。",
      f"总耗时 {(time.perf_counter()-t_all)/60:.1f} 分钟。"]
with open(os.path.join(ROOT, "docs", "nc_audit_v0.md"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(md))
with open(os.path.join(ROOT, "docs", "nc_audit_v0.json"), "w",
          encoding="utf-8") as f:
    json.dump(dict(summary=summ, rows=jout), f, ensure_ascii=False, indent=1)
print("→ docs/nc_audit_v0.md / .json")
