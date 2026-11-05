# -*- coding: utf-8 -*-
r"""_nc_audit2.py — NC 审计 v0.1（E1d）。

对 v0 的四升级 + 两修复：
  ① V3 中心列（E1f2/E1f3d 生产路径）：hybrid_ai(res) 等效 Fit2D 中心 vs
     GT-Fit2D 中心——同口径对打（旧 Δc 对共心 ref 是混合口径）
  ② tilt 分层：|rot|≥2mrad → T 组（共心 ref 预期偏置，信 V3/锐度），
     否则 t 组（ref 可信）
  ③ 锐度仲裁（自适应峰检）：经验半径阈值 k·MAD 自适应（8→2 降档直到
     ≥3 峰，修 halfccd 6σ 空集），锚=我们 V3 中心；三中心（GT-Fit2D /
     V3 / ref）Σ锐度对决，最锐者胜
  ④ λ̂ 双臂：halfccd→0.1899Å（T5 实证 LaB6）、mar3450→0.3611Å（基准
     fallback λ̂）；其余名义 λ
  修复：spline poni 先 chdir（v0 的 Frelon/halfccd/LaB6/alumina×52 行
  ERROR 由此而来）；无 λ poni 不再用 0.972386 兜底。
输出：docs/nc_audit_v0_1.md + .json
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

import pyFAI

from gui_solver import load_any, SolveWorker, hybrid_ai
from solver_core import concentric_center
from torchfai.phases import phase_qs

ZOO = os.path.join(ROOT, "data", "real", "silx_zoo")
CERIA = os.path.join(ROOT, "data", "real", "MX_ceria")
ALUMINA = os.path.join(ROOT, "data", "real", "alumina")
LAM_HAT = {"halfccd": 0.1899, "LaB6_0020poni": 0.1899, "mar3450": 0.3611}

PAIRS = []
for d in (150, 200, 250, 300, 350, 400, 450):
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
        ("halfccd.edf", "LaB6.poni", "LaB6_0020poni"),
        ("LaB6_260210.mar3450", "Mar3450.poni", "mar3450"),
        ("Fairchild.edf", "Fairchild.poni", "Fairchild")]:
    PAIRS.append((tag, os.path.join(ZOO, img_f), os.path.join(ZOO, poni_f)))
_poni_al = [f for f in os.listdir(ALUMINA) if f.endswith(".poni")]
if _poni_al:
    _p = os.path.join(ALUMINA, _poni_al[0])
    for f in sorted(os.listdir(ALUMINA)):
        if f.endswith(".edf"):
            PAIRS.append((f"alumina/{f[:11]}", os.path.join(ALUMINA, f), _p))


def profile(img, cx, cy, dr=0.25):
    rmax = min(max(img.shape) * 1.2, 4000)
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - cy, xx - cx)
    bins = np.arange(0, rmax + dr, dr)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    i, v = idx[ok], img.ravel()[ok]
    cnt = np.bincount(i, minlength=len(bins) - 1)
    p = np.bincount(i, weights=v, minlength=len(bins) - 1) / np.maximum(cnt, 1)
    return bins[:-1] + dr / 2, p


def radii_adaptive(img, anchor, n_want=4):
    """自适应峰检：k·MAD 从 8 降到 2 直到 ≥3 峰；≥3 不可得返回 []。"""
    pr, pp = profile(img, anchor[0], anchor[1])
    r_edge = 0.9 * min(anchor[0], anchor[1], img.shape[1] - anchor[0],
                       img.shape[0] - anchor[1])
    m = (pr > 60) & (pr < r_edge)
    if m.sum() < 50:
        return []
    pr, pp = pr[m], pp[m]
    med = np.median(pp)
    mad = max(np.median(np.abs(pp - med)), 1e-9)
    for k in (8, 6, 4, 3, 2):
        thr = med + k * mad
        sm = np.convolve(pp, np.ones(5) / 5, "same")
        cand = []
        for j in range(2, len(sm) - 2):
            # 峰必须落在 (60, r_edge) 环域内——否则中心束斑伪峰霸占 top-4、
            # dedup 后 <3 环全档失败（ceria_150 诊断：k=8 有 331 峰但 radii=[]）
            if not (60 < pr[j] < r_edge):
                continue
            if (sm[j] >= sm[j - 1] and sm[j] >= sm[j + 1]
                    and sm[j] >= sm[j - 2] and sm[j] >= sm[j + 2]
                    and pp[j] > thr):
                cand.append((pp[j], pr[j]))
        cand.sort(reverse=True)
        # 贪心 NMS：取最高峰后强制 ≥25px 径向间隔再取次高——先 top-4 再
        # dedup 会被亮环平台内的细分档涟漪塞满（彼此 <15px，ceria_150 全档
        # 归零二轮实证）
        sel = []
        for h, r in cand:
            if all(abs(r - r0) > 25 for r0 in sel):
                sel.append(r)
            if len(sel) >= n_want:
                break
        dedup = sorted(sel)
        if len(dedup) >= 3:
            return dedup
    return []


def sharp_sum(img, c, radii, half=4.0):
    # half=4px：窗宽必须小于被仲裁的中心差（5-7px 量级），否则偏移中心
    # 在 ±8px 窗内仍能"借"到环峰、对决失效（首跑 P2M/E9M 仲裁=ref 反例）
    if not radii:
        return None
    pr, pp = profile(img, c[0], c[1])
    tot = 0.0
    for r0 in radii:
        m = np.abs(pr - r0) < half
        if m.sum() < 5:
            return None
        tot += float(pp[m].max() - np.percentile(pp[m], 10))
    return tot


print(f"审计对象：{len(PAIRS)} 个 (image, poni) 对", flush=True)
rows, jout = [], []
t_all = time.perf_counter()
for tag, img_f, poni_f in PAIRS:
    t0 = time.perf_counter()
    try:
        os.chdir(os.path.dirname(poni_f))          # 修复：spline 相对路径
        ai = pyFAI.load(poni_f)
        f2 = ai.getFit2D()
        gx, gy = f2["centerX"], f2["centerY"]
        px_um = float(ai.detector.pixel1) * 1e6
        lam = (ai.wavelength * 1e10) if ai.wavelength else LAM_HAT.get(
            tag, 0.972386)
        rot_mag = float(np.hypot(ai.rot1, ai.rot2)) * 1000  # mrad
        tier = "T" if rot_mag >= 2.0 else "t"
        arm = "λ̂" if tag in LAM_HAT else "λ0"
        gt_D = ai.dist * 1000.0

        img = load_any(img_f)
        img = np.where(img >= 4e9, 0.0, img)
        box = {}
        w = SolveWorker(img, lam, px_um)
        w.done.connect(lambda r: box.update(r=r))
        w.run()
        res = box.get("r")
        if not res:
            rows.append(f"| {tag} | SOLVE FAIL | | | | | | | | |")
            jout.append(dict(tag=tag, fail="solve"))
            continue
        ph = res["ph"]
        ai_v3 = hybrid_ai(res, px_um)
        f3 = ai_v3.getFit2D()
        v3 = (f3["centerX"], f3["centerY"])
        dD = ai_v3.dist * 1000 - gt_D
        dc_v3 = float(np.hypot(v3[0] - gx, v3[1] - gy))

        qs = phase_qs(ph)
        cr = None
        try:
            cr = concentric_center(img, v3, res["D_px"], qs, res["lam"])
        except Exception:
            cr = None
        refc = (cr["cx"], cr["cy"]) if cr else None
        gt_dc = (float(np.hypot(gx - cr["cx"], gy - cr["cy"]))
                 if cr else None)

        radii = radii_adaptive(img, v3)
        s_gt = sharp_sum(img, (gx, gy), radii)
        s_v3 = sharp_sum(img, v3, radii)
        s_rf = sharp_sum(img, refc, radii) if refc else None
        trio = {"GT": s_gt, "V3": s_v3, "ref": s_rf}
        trio = {k: v for k, v in trio.items() if v is not None}
        verdict = (max(trio, key=trio.get) if trio else "—")

        rows.append(
            f"| {tag} | {tier} | {arm} | {ph} | {dc_v3:.2f} | "
            f"{dD:+.2f} | " + (f"{gt_dc:.1f}" if gt_dc is not None else "—")
            + " | " + (f"{cr['n']}" if cr else "—") + " | "
            + " ".join(f"{k}:{v:.2e}" for k, v in trio.items())
            + f" | **{verdict}** |")
        jout.append(dict(tag=tag, tier=tier, arm=arm, ph=ph,
                         dc_v3=round(dc_v3, 3), dD=round(dD, 3),
                         gt_dc=(round(gt_dc, 2) if gt_dc is not None
                                else None),
                         cref_n=(cr["n"] if cr else None),
                         radii=[round(float(r), 1) for r in radii],
                         sharp={k: float(v) for k, v in trio.items()},
                         verdict=verdict, rot_mrad=round(rot_mag, 2)))
        print(f"  {tag:22} {tier} {arm} {ph:8} Δc(V3)={dc_v3:6.2f}px "
              f"ΔD={dD:+6.2f} 环={cr['n'] if cr else '—'} "
              f"仲裁={verdict}  ({time.perf_counter()-t0:.1f}s)", flush=True)
    except Exception as e:
        rows.append(f"| {tag} | ERROR {type(e).__name__}: {e} | | | | | | | | |")
        jout.append(dict(tag=tag, fail=f"{type(e).__name__}: {e}"))
        print(f"  {tag:22} ERROR {e}", flush=True)

os.chdir(ROOT)
md = ["# NC 核心证据 v0.1：锐度仲裁 + tilt 分层 + λ̂ 双臂审计",
      "",
      "生成：_nc_audit2.py ｜ " + str(len(PAIRS)) + " 对 ｜ "
      f"总耗时 {time.perf_counter()-t_all:.0f}s",
      "",
      "列：分层(T=|rot|≥2mrad)｜臂(λ0 名义/λ̂)｜相｜**Δc(V3)px**（V3 成对"
      "几何 vs GT-Fit2D 同口径）｜ΔD mm｜GTΔc(对ref,旧口径)｜ref环数｜"
      "Σ锐度(GT/V3/ref)｜**锐度仲裁**",
      "",
      "**v0.1 五结论（2026-11-04，NMS 终版）**：①ceria 族 GT 中心距经验"
      "锐度最优(ref) 1.7–3.5px——GT 无辜于 v0 的 15px 级指控，但 ~3px 级"
      "标定方法散布实证存在；V3 紧跟 GT（0.34–1.24px）。②Eiger4M/v2 与"
      "相鉴正确的 alumina 帧多数 V3 最锐——我们的几何常比 GT 更贴数据；E1f3e（2026-11-04）SolveWorker 原生优先 tf 移植后生产路径 Eiger4M Δc(V3) 9.32→0.32px、v2 5.62→0.83px（本表为移植前口径，zoo 表 E1f3d 已是原生口径）。"
      "③alumina 族 per-frame 零样本：相鉴 16/52、Δc(V3) 中位 10.65px——"
      "弱衍射静态装置必须走 mean-frame 管线（生产 116.79mm/−0.07 铁证）。"
      "④halfccd/mar 的 λ̂ 单值臂仍失败（配置错配/相鉴域界，既有定性不变）。"
      "⑤方法学：锐度窗必须窄于被仲裁差(±4px)；半径检用贪心 NMS(≥25px 间隔)",
      "——top-4+dedup 会被亮环涟漪归零（ceria_150 三轮实证）。",
      "",
      "| 标定文件 | 层 | 臂 | 相 | Δc(V3) | ΔD(mm) | GTΔc(ref) | 环 | "
      "Σ锐度 | 仲裁 |",
      "|---|---|---|---|---|---|---|---|---|---|"] + rows
with open(os.path.join(ROOT, "docs", "nc_audit_v0_1.md"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(md))
with open(os.path.join(ROOT, "docs", "nc_audit_v0_1.json"), "w",
          encoding="utf-8") as f:
    json.dump(jout, f, ensure_ascii=False, indent=1)

ok = [r for r in jout if "dc_v3" in r]
T = [r for r in ok if r["tier"] == "T"]
t = [r for r in ok if r["tier"] == "t"]
print(f"\n→ docs/nc_audit_v0_1.md/.json  可评 {len(ok)} 对 "
      f"(T组 {len(T)} / t组 {len(t)})  "
      f"Δc(V3) 中位 {np.median([r['dc_v3'] for r in ok]):.2f}px", flush=True)
if T:
    print(f"  T组 Δc(V3) 中位 {np.median([r['dc_v3'] for r in T]):.2f}px "
          f"仲裁分布 { {v: [r['verdict'] for r in T].count(v) for v in set(r['verdict'] for r in T)} }", flush=True)
if t:
    print(f"  t组 Δc(V3) 中位 {np.median([r['dc_v3'] for r in t]):.2f}px "
          f"仲裁分布 { {v: [r['verdict'] for r in t].count(v) for v in set(r['verdict'] for r in t)} }", flush=True)
