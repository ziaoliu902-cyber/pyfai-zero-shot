# -*- coding: utf-8 -*-
"""八件套逐一消融 v3（真实数据：ceria 7 档 + Pilatus2M + Eiger4M）。

v3（2026-11-05，18-1104 需求）：全部臂改调 solver_core.solve_image 统一
入口（R1 画布默认档 520 + R2 零样本安全 fallback）——v2 的 canvas_of(723)
+ 裸 ps<0.55 fallback 门是 P2M −12.39 / E4M +244.7 的根因，已废。
A 臂应与 zoo_benchmark_v3 四胜行一致（P2M −0.05 / E4M −0.03）。

八件：
  A 完整        B 关 2D 精化     C 关自动定心     D 关级联
  E 喂 GT 相    F λ 未知         G 加倾角层       H NN 兜底（N/A）
"""
import os
import sys
import time

import numpy as np

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")

from torchfai.phases import phase_qs
from solver_core import solve_image
from poni_out import fit_tilt

OUT = os.path.join(ROOT, "probes", "results")
os.makedirs(OUT, exist_ok=True)


def datasets():
    ds = []
    for d in (150, 200, 250, 300, 350, 400, 450):
        ds.append(dict(tag=f"ceria_{d}",
                       img=os.path.join(ROOT, "data", "real", "MX_ceria",
                                        f"ceria_{d}_1_0001.cbf"),
                       poni=os.path.join(ROOT, "data", "real", "MX_ceria",
                                         f"ceria_{d}_1_0001.poni"),
                       ph="CeO2"))
    z = os.path.join(ROOT, "data", "real", "silx_zoo")
    ds.append(dict(tag="Pilatus2M", img=os.path.join(z, "Pilatus2M.cbf"),
                   poni=os.path.join(z, "Pilatus2M.poni"), ph="CeO2"))
    ds.append(dict(tag="Eiger4M", img=os.path.join(z, "Eiger4M.edf"),
                   poni=os.path.join(z, "Eiger4M.poni"), ph="Al2O3"))
    return ds


def load(img_path):
    """与 zoo_run.load_img 同口径：float64 + 哨兵(≥1e9)清零 + 负值清零。

    2026-11-05 修：v2/v3 的裸 float32 不清哨兵——Eiger4M.edf 脏像素把
    默认档解毁掉 → c-risk 分叉预检假阳性 → 救援采用 +244.7 垃圾解
    （与 zoo 同图同算法必须同答案，T7 哨兵教训的镜像案例）。"""
    import fabio
    try:
        arr = fabio.open(img_path).data
    except Exception:
        from PIL import Image
        arr = np.asarray(Image.open(img_path))
    img = np.asarray(arr, dtype=np.float64)
    img[img >= 1e9] = 0.0
    img[img < 0] = 0.0
    return img


def solve_arm(img, lam_A, px_native, ph_gt, arm):
    """单臂求解：solve_image 统一入口，arm 只改 solve_kw/phases/lam。"""
    t0 = time.time()
    kw = dict(cascade=True, auto_center=True, refine_2d=True)
    if arm == "B":
        kw["refine_2d"] = False
    elif arm == "C":
        kw["auto_center"] = False
    elif arm == "D":
        kw["cascade"] = False
    phases = [ph_gt] if arm == "E" else None
    if arm == "F":
        res = solve_image(img, px_native, lam=None,
                          solve_kw=dict(auto_center=True))
    else:
        res = solve_image(img, px_native, lam=lam_A, phases=phases,
                          solve_kw=kw)
    if res is None:
        return dict(ok=False, dt=time.time() - t0)
    ph, sol = res["phase"], res["sol"]
    s, px_m, canvas = res["s"], res["px_m"], res["canvas"]
    lam_used = res["lam"]

    cx, cy, D = sol[0] * s, sol[1] * s, sol[2] * px_m
    if arm == "G":
        try:
            qs = np.asarray(phase_qs(ph), float)
            tf = fit_tilt(canvas, sol[0], sol[1], sol[2], qs, lam_used,
                          px_m, px_m, sfull=s, nsec=36, mass_rel=0.10, bump=1.5)
            if tf is not None:
                cx, cy = tf["poni2"] / px_native, tf["poni1"] / px_native
        except Exception:
            pass
    return dict(ph=ph, cx=cx, cy=cy, D=D, ok=True, dt=time.time() - t0,
                used_fallback=res["used_fallback"],
                fb_reason=res["fb_reason"], s=s)


def main():
    import pyFAI
    arms = ["A", "B", "C", "D", "E", "F", "G"]
    per = {a: [] for a in arms}
    for d in datasets():
        img = load(d["img"])
        gt = pyFAI.load(d["poni"])
        px_nat = float(gt.detector.pixel1)
        lam_A = float(gt.wavelength) * 1e10
        gcx = gt.poni2 / px_nat - 0.5
        gcy = gt.poni1 / px_nat - 0.5
        for a in arms:
            r = solve_arm(img, lam_A, px_nat, d["ph"], a)
            if not r.get("ok"):
                per[a].append(dict(tag=d["tag"], ok=False)); continue
            per[a].append(dict(tag=d["tag"], ok=True,
                               dD=(r["D"] - gt.dist) * 1000,
                               dc=float(np.hypot(r["cx"] - gcx, r["cy"] - gcy)),
                               ph_ok=(r["ph"] == d["ph"]), dt=r["dt"],
                               fb=r.get("used_fallback"),
                               fbr=r.get("fb_reason"), s=r.get("s")))
        nfb = sum(1 for a in arms if per[a][-1].get("fb"))
        print(f"{d['tag']:<12}" + " ".join(
            f" {a}:" + (f"{per[a][-1]['dD']:+6.2f}/{per[a][-1]['dc']:5.1f}"
                        if per[a][-1].get("ok") else "FAIL")
            for a in arms) + (f"  [fallback×{nfb}]" if nfb else ""),
            flush=True)

    def agg(a):
        rs = [r for r in per[a] if r.get("ok")]
        if not rs:
            return {}
        dD = np.abs([r["dD"] for r in rs]); dc = np.array([r["dc"] for r in rs])
        return dict(n=len(rs), dD=float(np.median(dD)), dDmax=float(dD.max()),
                    dc=float(np.median(dc)), dcmax=float(dc.max()),
                    p=float(np.mean([r["ph_ok"] for r in rs])),
                    dt=float(np.mean([r["dt"] for r in rs])))

    md = ["# 八件套消融 v3（真实数据，solve_image 统一入口）", "",
          "v3（2026-11-05）：全部臂走 solver_core.solve_image（R1 默认档 520",
          "+ R2 零样本安全 fallback），A 臂口径= zoo_benchmark_v3。",
          "",
          "| 臂 | 说明 | 解出 | 中位\\|ΔD\\| | 最差\\|ΔD\\| | 中位Δc | 最差Δc | 相正确 | 耗时 |",
          "|---|---|---|---|---|---|---|---|---|"]
    names = dict(A="完整", B="关 2D 精化", C="关自动定心", D="关级联",
                 E="喂 GT 相", F="λ 未知", G="加倾角层")
    for a in arms:
        g = agg(a)
        if not g:
            md.append(f"| {a} | {names[a]} | 0 | — | — | — | — | — | — |"); continue
        md.append(f"| {a} | {names[a]} | {g['n']}/9 | {g['dD']:.2f} | {g['dDmax']:.2f} | "
                  f"{g['dc']:.2f} | {g['dcmax']:.1f} | {g['p']*100:.0f}% | {g['dt']:.1f}s |")
    md.append("| H | NN 兜底 | — | 经典路径不含 NN（classic-first），N/A | | | | | |")
    md += ["", "## 逐数据集", ""]
    for i, d in enumerate(datasets()):
        cells = []
        for a in arms:
            r = per[a][i]
            c = f"{a}:" + (f"{r['dD']:+.2f}/{r['dc']:.1f}px" if r.get("ok")
                           else "FAIL")
            if r.get("fb"):
                c += f"[fb:{r.get('fbr')},s={r.get('s')}]"
            cells.append(c)
        md.append(f"- **{d['tag']}**: " + " | ".join(cells))
    ts = time.strftime("%Y%m%d_%H%M")
    p = os.path.join(OUT, f"ablation8v3_{ts}.md")
    open(p, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md)); print("saved", p)


if __name__ == "__main__":
    main()
