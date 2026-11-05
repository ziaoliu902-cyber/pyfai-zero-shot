# -*- coding: utf-8 -*-
"""gui_solver.py：衍射图盲解 GUI（2026-11-02，T-GUI）。

 PyQt5 + matplotlib 零新依赖。功能：
 · 打开图像（edf/cbf/h5/mar/tif/png——zoo load_img 优先，PIL 兜底）；
 · 可选波长输入（用户 λ 破 λ–D 简并——小角数据必填才出物理 D）；
 · 后台线程跑 solver_core 全链（画布解 → 原生 2D 精化），界面不卡；
 · 图上叠加：物相预测环系（绿）+ 束心十字（红）+ 2D 精化控制环（黄）；
 · 右栏：相鉴定/λ̂/D/中心/m/rms/ps + 径向谱（相线标记）；
 · 导出 .poni（pyFAI 直读；rot=0 直立几何，注释写明口径）。

 用法：python gui_solver.py
"""
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))  # 生产包根（torchfai 等）

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit,
    QProgressBar, QMessageBox, QGroupBox, QFormLayout)

import matplotlib
matplotlib.use("QtAgg")
# 中文字形：DejaVu Sans 无 CJK → 标题会渲染成 □ 豆腐块。
# Windows 用微软雅黑兜底黑体，负号也交给 CJK 字体（unicode_minus）。
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                          "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from solver_core import (to_canvas, solve_image, ring_radii,
                         refine_geometry_2d, _r_model)
from torchfai.phases import phase_qs
from i18n import tr


def apply_spline(img, sp_path):
    """畸变样条应用（T-SPL，2026-11-03）：加载 .spline → 反畸变 → 边带置零。

    - 正逆方向由 solver_core.undistort 负责（不动点逆映射，往返 0.05px 实证，
      切忌"直接减位移"——那是正向场，±号会互相矛盾）；
    - 边带置零（F5）：undistort 用 mode="nearest" 重采样，四边是钳位外推的
      涂抹带，不置零会被找峰当成环。带宽 = 3×样条控制点间距（pyFAI Spline
      的 splineX 节点差），取不到节点则回退 16px，夹在 [8, 64]。
    - 峰位影响：双线性插值让峰轻微变宽（σ ~+0.2px 级），验证 A 卡 0.1mm
      裕量足够；不够再升 order=3。
    """
    from pyFAI.spline import Spline
    from solver_core import undistort
    sp = Spline(sp_path)
    out = undistort(np.asarray(img, float), sp)
    m = 16
    try:
        kx = np.asarray(sp.splineX, float)
        m = int(round(3.0 * np.median(np.diff(np.unique(kx)))))
    except Exception:
        pass
    m = int(np.clip(m, 8, 64))
    out[:m, :] = out[-m:, :] = 0.0
    out[:, :m] = out[:, -m:] = 0.0
    return out


def set_splinefile(ai, path):
    """F4 兼容 setter：pyFAI≥2025.09 属性名小写 splinefile（CamelCase 仍可读
    但每次触碰都在 stderr 刷 DeprecationWarning）；旧版回退 splineFile。"""
    d = ai.detector
    if isinstance(getattr(type(d), "splinefile", None), property):
        d.splinefile = path
    else:                                   # 旧版 pyFAI：CamelCase 赋值即可
        d.splineFile = path


def load_any(path):
    """edf/cbf/mar/tif → fabio；h5 → h5py 找 2D data；png/jpg → PIL。

    不再 import zoo_run——它模块级 import pyFAI.azimuthalIntegrator
    （pyFAI≥2024.10 弃用路径，会抛 DeprecationError），GUI 自带轻载。"""
    try:
        import fabio
        return np.asarray(fabio.open(path).data, float)
    except Exception:
        pass
    if path.lower().endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(path, "r") as f:
            hit = []

            def _walk(name, obj):
                if not hit and isinstance(obj, h5py.Dataset) \
                        and obj.ndim >= 2 and name.split("/")[-1] == "data":
                    hit.append(obj)

            f.visititems(_walk)
            if not hit:
                raise IOError("h5 里找不到 2D 'data' 数据集")
            d = hit[0]
            return np.asarray(d[0] if d.ndim > 2 else d[()], float)
    from PIL import Image
    return np.asarray(Image.open(path), float)


def sniff_meta(path):
    """从文件头偷元数据（像素自动读取，三级：头正则→尺寸指纹→手填）。

    ESRF/Dectris CBF 头全有（实测 ceria_400：Pixel_size/Wavelength/
    Detector_distance/Beam_xy）。求解器不用这些值（盲解口径）——只做
    输入框预填与对照参考。"""
    out, src = {}, []
    try:
        import re
        import fabio
        im = fabio.open(path)
        blob = str(im.header.get("_array_data.header_contents", ""))
        flat = "\n".join(f"{k}: {v}" for k, v in im.header.items())
        text = blob + "\n" + flat
        m = re.search(r"Pixel_size\s+([\d.eE+-]+)\s*m\s*x", text)
        if m:
            out["px"] = float(m.group(1)) * 1e6
            src.append("头·Pixel_size")
        m = re.search(r"Wavelength\s+([\d.]+)\s*A", text)
        if m:
            out["lam"] = float(m.group(1))
            src.append("头·Wavelength")
        m = re.search(r"Detector_distance\s+([\d.]+)\s*m", text)
        if m:
            out["D_mm"] = float(m.group(1)) * 1000
            src.append("头·Detector_distance")
        m = re.search(r"Beam_xy\s+\(([\d.]+),\s*([\d.]+)\)", text)
        if m:
            out["beam"] = (float(m.group(1)), float(m.group(2)))
            src.append("头·Beam_xy")
        if "px" not in out:
            fp = {(1043, 981): 172, (1679, 1475): 172, (2527, 2463): 172,
                  (2167, 2070): 75, (3262, 3108): 75,
                  (2048, 2048): 47, (1024, 2048): 47.6}
            key = tuple(int(v) for v in im.data.shape)
            if key in fp:
                out["px"] = fp[key]
                src.append("尺寸指纹%s" % (key,))
    except Exception:
        pass
    out["src"] = " + ".join(src)
    return out


def hybrid_ai(res, px_um):
    """几何合成 → AzimuthalIntegrator（自检/导出/向导三处同源）。

    T7 接受：T7 中心+D（r2d 挣的精度不丢）；倾角仅在 tilt_on（改善≥5%）
    时叠加——小角数据倾角不可辨识，硬加会把 0.010° 吃成 0.138°（P2M
    实证）。T7 拒收：tf 的 poni+rot（PONI 与 rot 成对自洽）+ 画布 D
    （ceria 0.024° 路径）。
    E1f2 V3（2026-11-03）：tilt_on（倾角被 imp≥5% 采纳或非 native 无条件）
    时升级为成对几何——D 仍用解算值（ΔD 回归零风险），poni/rot 用 tf 的
    自洽对。判决 _nc_e1f2_ab.py / _nc_e1f3b.py：ceria_150/250/300 dev
    0.166/0.115/0.188° → 0.045/0.046/0.041°，Σ锐度 10–67×，中心
    6.7/5.1/25.2px → 1.2/0.3/0.5px；非 native 同样适用——ceria_400
    0.095°→0.024°/Σ17×/中心 24.5→2.5px、Eiger4M Σ7×/中心 42→9px；
    E9M 中心 0.66px/锐度 87% GT。"只借 rot"（V1）实证 10× 更钝，
    全面禁用。P2M 负对照：imp=1% 门关，倾角不硬加（小角域教训闸门）。"""
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    px_m = px_um * 1e-6
    tf = res.get("tf") if res.get("tilt_on") else None
    if tf is not None:
        return AzimuthalIntegrator(dist=res["D_px"] * px_m,
                                   poni1=tf["poni1"], poni2=tf["poni2"],
                                   pixel1=px_m, pixel2=px_m,
                                   rot1=tf["rot1"], rot2=tf["rot2"], rot3=0.,
                                   wavelength=res["lam"] * 1e-10)
    # 中心用 Worker 选定的那个（r2d / fit_tilt 中环 rms 更小者）；倾角只借 rot。
    return AzimuthalIntegrator(dist=res["D_px"] * px_m,
                               poni1=res["cy"] * px_m,
                               poni2=res["cx"] * px_m,
                               pixel1=px_m, pixel2=px_m,
                               rot1=(tf["rot1"] if tf else 0.0),
                               rot2=(tf["rot2"] if tf else 0.0), rot3=0.,
                               wavelength=res["lam"] * 1e-10)


def check_std(img, res, px_um):
    """标样自检：解出的 poni 积分 1D → 自动找峰 → 对相库答案卡。

    返回 dict(lines=[...], n, dev_max, verdict)。判据 |Δ2θ|<0.01°。
    预测与积分两侧同用 λ̂ → 检验的是几何+相的整体自洽（正是
    交付物峰位的误差）。"""
    import numpy as np
    lam = res["lam"]
    # 几何与导出/向导同源：一律走 hybrid_ai（含 tilt_on 门控——T7 接受
    # 且倾角改善<5% 时 rot 不叠加；P2M 实证：老规则"有 tf 就用"会把
    # T7 的 0.010° 吃成 0.156°）
    ai = hybrid_ai(res, px_um)
    tth, I = ai.integrate1d(img, 16000, unit="2th_deg",
                            method="bbox,splitpixel")
    med = float(np.median(I))
    thr = med + 0.05 * (I.max() - med)   # 判"有峰"的最弱信噪门
    step = tth[1] - tth[0]
    H, W = img.shape
    lines, devs, pairs = [], [], []
    qs = phase_qs(res["ph"])
    for i, q in enumerate(qs, 1):
        r = _r_model(q, lam, res["D_px"])
        if not (3 < r < 1.3 * max(H, W)):
            continue  # 环在画面外，不参评
        t_pred = float(np.degrees(2 * np.arcsin(q * lam / (4 * np.pi))))
        if t_pred > tth[-1] - 0.5 or t_pred < tth[0] + 0.5:
            continue
        # 答案卡在先：预测位 ±0.2° 窗内取最大 + 抛物线内插。窗口宽窄的
        # 两难（P2M 乌龙 2026-11-02）：太宽（0.35°）会把 0.21° 外的邻居
        # 强环抓来冒名弱环→假 0.28° 偏差（GT poni 也被误判）；太窄会漏
        # D 误差 0.35% 以上的真峰。0.2° = 两者折中，ceria/P2M 双实证。
        win = (tth > t_pred - 0.20) & (tth < t_pred + 0.20)
        wt, wi = tth[win], I[win]
        if wt.size < 5 or wi.max() <= thr:
            lines.append(tr("ck_nopeak", i=i, t=t_pred))
            continue
        k = int(np.argmax(wi))
        if k < 4 or k > len(wi) - 5:
            # 极值贴窗沿 = 邻峰尾巴伸进来（P2M #8 实证：Δ=−0.200 恰=窗宽）
            lines.append(tr("ck_nopeak2", i=i, t=t_pred))
            continue
        t_meas = wt[k]
        if 0 < k < len(wi) - 1:
            y0, y1, y2 = wi[k - 1], wi[k], wi[k + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-12:
                t_meas += float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) * step
        dev = t_meas - t_pred
        devs.append(abs(dev))
        pairs.append((t_pred, t_meas))
        lines.append(f"  #{i:2d}  {t_pred:8.3f}° vs {t_meas:8.3f}°"
                     f"  Δ{dev:+.3f}°")
    # ---- 精度评估：Δ2θ = −k·tan(2θ) 单参最小二乘 ----
    # 曾试两参（δD/D 走 tan2θ、d 标尺走 tanθ）——实测共线（P2M 列相关
    # 0.999997，ceria 五环也 0.9996），最小范数解拆成两个巨量对消的
    # 伪 11%/0.8%。无独立 λ 时 D 误差与 d 标尺本不可分，合并为单一
    # 标尺项 k 才是物理可观测。
    if len(pairs) >= 3:
        tp = np.radians([p[0] for p in pairs])
        y = np.radians([p[1] for p in pairs]) - tp        # Δ2θ (rad)
        tx = np.tan(tp)
        k = -float(tx @ y / (tx @ tx))          # 模型 y = −k·tan2θ，k ≙ δD/D
        rms = float(np.sqrt(np.mean((y + k * tx) ** 2)))
        D_mm = res["D_px"] * px_um * 1e-3
        lines.append(tr("prec_head"))
        lines.append(tr("prec_ang", v=f"{np.degrees(rms):.4f}", n=len(pairs)))
        lines.append(tr("prec_scale", p=f"{abs(k) * 100:.3f}",
                        mm=f"{abs(k) * D_mm:.2f}"))
        dd = rms / (2.0 * np.tan(tp.max() / 2.0))
        lines.append(tr("prec_d", p=f"{dd * 100:.3f}",
                        t=f"{np.degrees(tp.max()):.1f}"))
    n = len(devs)
    if n >= 3:
        dm = max(devs)
        vs = f"{dm:.3f}"
        if dm < 0.01:
            verdict = tr("v_perfect", v=vs)
        elif dm < 0.05:
            verdict = tr("v_pass", v=vs)
        elif dm < 0.2:
            verdict = tr("v_warn", v=vs)
        else:
            verdict = tr("v_bad", v=vs)
    else:
        verdict = tr("v_few")
    return dict(lines=lines, n=n, dev_max=(max(devs) if devs else None),
                verdict=verdict)


def refine_scale_1d(img, cx, cy, D0, qs, lam, rel=0.04, nb=3000):
    """1D 径向谱 D 尺度精化（GUI 兜底层，2026-11-02）。

    refine_geometry_2d 只收完整环（r<0.9×边距）——束心离边近的图
    （ceria/P6M 中心贴边）会饿死拒收，D 停在画布解的 ~%级误差。
    这里对完整环无要求：径向谱只算一次，D 网格上让相答案卡梳齿
    与谱峰对齐（对比局部本底），抛物线内插极值。中心沿用画布
    auto_center（1~6px 量级，对峰位二阶小）。
    返回 (D_px精化, score) 或 None。"""
    import numpy as np
    H, W = img.shape
    rmax = float(np.hypot(*[max(cx, W - 1 - cx), max(cy, H - 1 - cy)]))
    rmax = min(rmax, 1.05 * np.hypot(W, H) * 0.75)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot(xx - cx, yy - cy).ravel()
    v = np.asarray(img, np.float32).ravel()
    prof, _ = np.histogram(r, bins=nb, range=(0, rmax), weights=v)
    cnt, _ = np.histogram(r, bins=nb, range=(0, rmax))
    ok = cnt > 0
    prof = np.where(ok, prof / np.maximum(cnt, 1), np.nan)
    # 平滑 + NaN 插值（缝/画外）
    p = np.nan_to_num(prof, nan=0.0)
    p = np.convolve(np.pad(p, 1, mode="edge"), np.ones(3) / 3, "valid")
    step = rmax / nb
    tan = np.tan(2 * np.arcsin(np.asarray(qs, float) * lam / (4 * np.pi)))
    tan = tan[(tan * D0 > 4) & (tan * D0 < rmax * 0.98)]
    if tan.size < 3:
        return None
    Dg = D0 * (1 + np.linspace(-rel, rel, 161))
    scores = []
    for D in Dg:
        ri = tan * D
        b = ri / step
        lo, hi = np.clip(b - 8, 0, nb - 1).astype(int), np.clip(b + 8, 0, nb - 1).astype(int)
        ri_b = np.clip(b, 0, nb - 1).astype(int)
        scores.append(np.mean(p[ri_b] - 0.5 * (p[lo] + p[hi])))
    scores = np.asarray(scores)
    k = int(np.argmax(scores))
    if not (0 < k < len(scores) - 1):
        return None
    y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    d = float(np.clip(d, -1, 1))
    Df = Dg[k] + d * (Dg[1] - Dg[0])
    return float(Df), float(scores[k])


class CheckWorker(QThread):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)

    def __init__(self, img, res, px_um):
        super().__init__()
        self.img, self.res, self.px_um = img, res, px_um

    def run(self):
        try:
            self.done.emit(check_std(self.img, self.res, self.px_um))
        except Exception:
            import traceback
            self.fail.emit(traceback.format_exc(limit=6))


class SolveWorker(QThread):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)

    def __init__(self, img, lam, px_um):
        super().__init__()
        self.img, self.lam, self.px_um = img, lam, px_um

    def run(self):
        try:
            img = self.img
            # ---- R3 统一入口（18-1104）：画布策略与 fallback 门控全部
            # 收敛到 solver_core.solve_image（默认档 520；零样本安全门
            # a/b/c 触发才升级大画布）。旧直调 solve_auto/solve_auto_lambda
            # + to_canvas 已废——同一算法 GUI 与基准必须同一答案。----
            res = solve_image(img, self.px_um * 1e-6,
                              lam=(float(self.lam) if self.lam else None))
            if res is None:
                self.done.emit(None)
                return
            canvas, s = res["canvas"], res["s"]
            px_m = self.px_um * 1e-6 * s
            ph, sol, lam = res["phase"], res["sol"], res["lam"]
            # T7 原生 2D 精化
            qs = np.asarray(phase_qs(ph), float)
            out = refine_geometry_2d(img, sol[0] * s, sol[1] * s,
                                     sol[2] * s, qs, lam, n_iter=3)
            native = False
            if out is not None:
                cx, cy, D_px, nr = out
                native = True
            else:
                cx, cy, D_px, nr = (sol[0] * s, sol[1] * s,
                                    sol[2] * s, 0)
            # 倾角层（fit_tilt 控制点 5 参）。种子策略与门控（P2M 实证
            # 2026-11-02）：T7 接受时用 T7 精化值做种子（rot 估计从
            # (+34,−7) 噪声底收敛到 (0,−2) mrad）；且倾角改善不足 5%
            # 时不叠加——小角数据倾角不可辨识，硬加反而把 T7 挣来的
            # 0.010° 吃成 0.138°。T7 拒收时倾角层是唯一中心/倾角来源，
            # 无条件上（ceria：0.35°→0.024°）。
            tf = None
            tilt_on = False
            # E1f2/E1f3e（2026-11-03/04）：倾角拟合原生优先。池化画布会把
            # imp 压到 <5% 错杀真倾角（ceria 画布 imp<5% vs 原生 34–39%，
            # _nc_e1f2_ab.py）；E1f3d zoo 实证非 native 帧同样受益——
            # Eiger4M canvas tf 中心 9.3px vs native 0.3px（rot 距 GT 从
            # ~12mrad 收到 ~1mrad）。native 拟合失败（弱环，ceria_400）→
            # 画布兜底。poni/rot 成对自洽由 hybrid_ai 的 V3 分支承接。
            tf_native = False
            try:
                from poni_out import fit_tilt as _ft
                # cx/cy/D_px 在两分支下均已是原生像素（T7 接受=精化值；
                # 拒收=sol*s 写回，见上）——直接用，勿再乘 s
                tf = _ft(img, cx, cy, D_px, qs, lam,
                         px_m / s, px_m / s, sfull=1, nsec=36,
                         mass_rel=0.10, bump=1.5)
                tf_native = True
                scx, scy, sDc, scv, sfs = cx, cy, D_px, img, 1
                px_fit = px_m / s
            except Exception:
                tf = None
            if tf is None and not native:
                try:      # 画布兜底（非 native 弱环帧，ceria_400 路径）
                    tf = _ft(canvas, sol[0], sol[1], sol[2], qs, lam,
                             px_m, px_m, sfull=s, nsec=36,
                             mass_rel=0.10, bump=1.5)
                    scx, scy, sDc, scv, sfs = (sol[0], sol[1], sol[2],
                                               canvas, s)
                    px_fit = px_m
                except Exception:
                    tf = None
            if tf is not None:
                tilt_on = (not native) or ((tf.get("imp") or 0) >= 0.05)
            # —— 交付中心：r2d 中心 vs fit_tilt 中心，取环一致性 rms 更小者 ——
            # 不依赖 GT："环圆不圆"是几何自洽的直接证据（GT 中心本身可能偏，
            # ceria 实测 GT 处环最糊：峰高 53/宽 18.6px vs 我们中心 169/13.1px）。
            ring_rms_v, ring_n, center_src = 0.0, 0, ("r2d" if native else "solver")
            try:
                from uncertainty import ring_rms
                r_s, n_s = ring_rms(img, cx, cy, D_px, qs, lam)
                r_t, n_t = 0.0, 0
                tcx = tcy = None
                if tf is not None:
                    px_n = self.px_um * 1e-6
                    tcx, tcy = tf["poni2"] / px_n, tf["poni1"] / px_n
                    r_t, n_t = ring_rms(img, tcx, tcy, D_px, qs, lam)
                if r_t > 0 and (r_s == 0 or r_t < r_s):
                    cx, cy = tcx, tcy
                    center_src = "tilt"
                    ring_rms_v, ring_n = r_t, n_t
                else:
                    ring_rms_v, ring_n = r_s, n_s
            except Exception:
                pass
            # —— 精度（不需要 GT 的经验估计，见 uncertainty.py）——
            sD_mm = 0.0
            s_r1 = s_r2 = 0.0
            try:
                from uncertainty import dist_sigma
                if native:
                    sD_px = dist_sigma(img, cx, cy, D_px, qs, lam,
                                       self.px_um * 1e-6)
                    sD_mm = sD_px * self.px_um * 1e-3
            except Exception:
                sD_mm = 0.0
            if tf is not None:
                try:
                    tf_lo = _ft(scv, scx, scy, sDc, qs, lam,
                                px_fit, px_fit, sfull=sfs, nsec=18,
                                mass_rel=0.10, bump=1.5)
                    if tf_lo is not None:
                        s_r1 = abs(tf_lo["rot1"] - tf["rot1"]) / np.sqrt(2)
                        s_r2 = abs(tf_lo["rot2"] - tf["rot2"]) / np.sqrt(2)
                except Exception:
                    pass
            self.done.emit(dict(ph=ph, sol=sol, lam=lam, s=s,
                                canvas=canvas, px_m=px_m,
                                used_fallback=res["used_fallback"],
                                fb_reason=res["fb_reason"],
                                cx=cx, cy=cy, D_px=D_px, n2d=nr,
                                native=native, tf=tf, tilt_on=tilt_on,
                                tf_native=tf_native,
                                sD_mm=sD_mm, s_r1=s_r1, s_r2=s_r2,
                                ring_rms=ring_rms_v, ring_n=ring_n,
                                center_src=center_src,
                                lam_free=not self.lam))
        except Exception:
            self.fail.emit(traceback.format_exc(limit=6))


class Win(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("衍射图盲解 · 相鉴定 + 几何标定")
        self.resize(1280, 800)
        self.img = None
        self.img_raw = None      # 畸变校正前原图（撤销用；None=未校正）
        self.spline = None       # 已应用的 .spline 路径（None=未校正）
        self.res = None
        self.worker = None
        self.poni_path = None
        self.st = QSettings("pyfai", "gui_solver")
        self.setAcceptDrops(True)          # 拖文件进窗口就能打开

        # ---- 三区布局（sci-desktop-ui spec §1）：左参数·中画布·右结果 ----
        import theme

        # 左：参数区（固定 ~300px）
        g_in = QGroupBox("输入")
        in_l = QVBoxLayout()
        self.b_open = QPushButton("📂 打开图像")
        self.b_open.clicked.connect(self.open_img)
        self.b_open.setToolTip("Ctrl+O；也可以把 edf/cbf/h5/mar/tif/png 直接拖进窗口")
        self.b_open.setProperty("primary", True)
        self.lbl_file = QLabel("未加载图像")
        self.lbl_file.setProperty("muted", True)
        self.lbl_file.setWordWrap(True)
        # 畸变校正（T-SPL F1/F2）：选 .spline → 反畸变 → 边带置零；可撤销
        self.b_sp = QPushButton("🧿 畸变校正…")
        self.b_sp.clicked.connect(self.apply_corr)
        self.b_sp.setEnabled(False)
        self.b_sp.setToolTip("选择探测器畸变样条 .spline，先校正回理想几何再盲解\n（FReLoN / 老 CCD 常见；2048² 约 1.5s，6M 约 3–5s）")
        self.b_sp_off = QPushButton("↩ 取消校正")
        self.b_sp_off.clicked.connect(self.undo_corr)
        self.b_sp_off.setVisible(False)
        self.sp_lab = QLabel("")
        self.sp_lab.setProperty("muted", True)
        self.sp_lab.setWordWrap(True)
        in_l.addWidget(self.b_open)
        in_l.addWidget(self.lbl_file)
        in_l.addWidget(self.b_sp)
        in_l.addWidget(self.b_sp_off)
        in_l.addWidget(self.sp_lab)
        g_in.setLayout(in_l)

        g_par = QGroupBox("参数")
        par_f = QFormLayout()
        self.e_px = QLineEdit("172"); self.e_px.setValidator(QDoubleValidator())
        self.e_px.setToolTip("探测器像素尺寸（µm）。文件头有就自动填；常见：Pilatus 172、Eiger 75、Frelon 47")
        self.e_lam = QLineEdit(""); self.e_lam.setValidator(QDoubleValidator())
        self.e_lam.setPlaceholderText("空 = 联立解")
        self.e_lam.setToolTip("波长（Å）。空着=盲解联立 λ–D；小角数据填了才出物理 D")
        h_px = QLabel("µm")
        h_px.setProperty("muted", True)
        par_f.addRow(QLabel("像素"), self.e_px)
        par_f.addRow(QLabel("波长 (Å)"), self.e_lam)
        g_par.setLayout(par_f)

        g_run = QGroupBox("运行")
        run_l = QVBoxLayout()
        self.b_run = QPushButton("▶ 运行求解")
        self.b_run.clicked.connect(self.run_solve)
        self.b_run.setEnabled(False)
        self.b_run.setToolTip("Ctrl+R")
        self.b_run.setProperty("primary", True)
        self.b_check = QPushButton("标样自检"); self.b_check.clicked.connect(self.run_check)
        self.b_check.setEnabled(False)
        self.b_check.setToolTip("用解出的 poni 积分→自动找峰→对答案卡，四级判语（Ctrl+K）")
        self.b_poni = QPushButton("导出 .poni"); self.b_poni.clicked.connect(self.export_poni)
        self.b_poni.setEnabled(False)
        self.b_copy = QPushButton("复制结果"); self.b_copy.clicked.connect(self.copy_result)
        self.b_copy.setEnabled(False)
        for b in (self.b_run, self.b_check, self.b_poni, self.b_copy):
            run_l.addWidget(b)
        g_run.setLayout(run_l)

        self.b_theme = QPushButton("🌙 暗色主题")
        self.b_theme.clicked.connect(self.toggle_theme)
        self.b_theme.setToolTip("Ctrl+T 切换亮/暗主题（matplotlib 画布同步）")
        left_l = QVBoxLayout()
        left_l.addWidget(g_in)
        left_l.addWidget(g_par)
        left_l.addWidget(g_run)
        left_l.addStretch(1)
        left_l.addWidget(self.b_theme)
        left_w = QWidget(); left_w.setLayout(left_l)
        left_w.setFixedWidth(300)

        # 中：画布（吃掉全部富余空间）
        self.fig = Figure(figsize=(8, 8))
        self.cv = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.tb = NavigationToolbar2QT(self.cv, self)   # 缩放/平移/存图
        self.tb.setStyleSheet("QToolBar { border: none; }")
        cen_l = QVBoxLayout()
        cen_l.setContentsMargins(0, 0, 0, 0)
        cen_l.addWidget(self.tb)
        cen_l.addWidget(self.cv, 1)
        cen_w = QWidget(); cen_w.setLayout(cen_l)

        # 右：结果（等宽字体）+ 径向谱
        self.txt = QTextEdit(); self.txt.setReadOnly(True)
        self.txt.setMinimumWidth(330)
        from PyQt5.QtGui import QFont
        self.txt.setFont(QFont(theme.MONO, 9))
        self.fig2 = Figure(figsize=(3.4, 2.6))
        self.cv2 = FigureCanvasQTAgg(self.fig2)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("径向谱（画布）", fontsize=9)
        right_l = QVBoxLayout()
        right_l.addWidget(self.txt, 3)
        right_l.addWidget(self.cv2, 2)
        right_w = QWidget(); right_w.setLayout(right_l)

        # ---- 底栏 ----
        self.pb = QProgressBar(); self.pb.setRange(0, 0); self.pb.setVisible(False)
        self.status = QLabel("就绪——打开图像（Ctrl+O）或把文件拖进来")
        self.status.setProperty("muted", True)
        self.status.setToolTip("结果可一键导出 .poni；右栏径向谱标出了物相预测的环位置")
        bot_l = QHBoxLayout()
        bot_l.addWidget(self.pb)
        bot_l.addWidget(self.status, 1)
        bot_w = QWidget(); bot_w.setLayout(bot_l)

        zones = QHBoxLayout()
        zones.addWidget(left_w)
        zones.addWidget(cen_w, 1)
        zones.addWidget(right_w)
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addLayout(zones, 1)
        outer.addWidget(bot_w)
        w = QWidget(); w.setLayout(outer); self.setCentralWidget(w)

        # 主题（记忆上次选择）
        self.theme_name = str(self.st.value("theme", "light"))
        self.apply_theme(self.theme_name)

        # 恢复上次的像素/波长（用户不用每次重填）
        try:
            px = self.st.value("px", "")
            lam = self.st.value("lam", "")
            if px:
                self.e_px.setText(str(px))
            if lam:
                self.e_lam.setText(str(lam))
        except Exception:
            pass
        # 快捷键
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtWidgets import QShortcut
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_img)
        QShortcut(QKeySequence("Ctrl+R"), self, self.run_solve)
        QShortcut(QKeySequence("Ctrl+E"), self, self.export_poni)
        QShortcut(QKeySequence("Ctrl+K"), self, self.run_check)
        QShortcut(QKeySequence("Ctrl+T"), self, self.toggle_theme)

    # ---------- 主题 ----------
    def apply_theme(self, name):
        import theme
        self.theme_name = name
        theme.apply_app(QApplication.instance(), name)
        theme.use_mpl(name)
        for ax in (self.ax, self.ax2):
            theme.restyle_axes(ax, name)
        self.cv.draw_idle()
        self.cv2.draw_idle()
        self.b_theme.setText("☀️ 亮色主题" if name == "dark" else "🌙 暗色主题")
        self.st.setValue("theme", name)

    def toggle_theme(self):
        self.apply_theme("dark" if self.theme_name == "light" else "light")

    # ---------- 拖拽打开 ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self._load(urls[0].toLocalFile())

    def copy_result(self):
        QApplication.clipboard().setText(self.txt.toPlainText())
        self.status.setText("结果已复制到剪贴板")

    # ---------- 畸变校正（T-SPL，2026-11-03 需求 F1/F2/F5） ----------
    def apply_corr(self):
        """选 .spline → 反畸变（不动点逆映射）→ 边带置零。再次选择=换样条，
        始终从原图重算，不叠加。校正图即 self.img → 求解/自检/导出全链
        自动走校正图（F3）。"""
        if self.img is None:
            return
        start = self.st.value("last_dir", "") or ""
        p, _ = QFileDialog.getOpenFileName(self, "选择畸变样条 .spline", start,
                                           "Spline (*.spline);;所有 (*.*)")
        if not p:
            return
        self.status.setText("畸变校正中——逆映射重采样（2048² 约 1.5s，6M 约 3–5s）…")
        QApplication.processEvents()          # 让状态栏先画出来
        try:
            base = self.img_raw if self.img_raw is not None else self.img
            self.img = apply_spline(base, p)
        except Exception as e:
            QMessageBox.warning(self, "校正失败", str(e))
            self.status.setText("畸变校正失败")
            return
        self.img_raw = base
        self.spline = p
        self._sp_refresh()
        self._invalidate_res("已应用校正")
        self.show_img()
        self.status.setText(f"已校正（{os.path.basename(p)}）· 边缘无效带已置零——请重新求解")

    def undo_corr(self):
        """撤销校正：回到原图，清 spline 态（poni 也不再写 SplineFile）。"""
        if self.img_raw is None:
            return
        self.img = self.img_raw
        self.img_raw = None
        self.spline = None
        self._sp_refresh()
        self._invalidate_res("已取消校正")
        self.show_img()
        self.status.setText("已取消校正，回到原始图")

    def _sp_refresh(self):
        """F2：校正状态可见——标签 + 撤销按钮显隐。"""
        on = self.spline is not None
        self.b_sp_off.setVisible(on)
        self.sp_lab.setText(
            f"已校正（{os.path.basename(self.spline)}）\n"
            f"求解/自检/导出均基于校正图\n导出 poni 将写入 SplineFile" if on else "")

    def _invalidate_res(self, why=""):
        """换图/校正后旧解作废——res 与 poni/自检/复制按钮同步失效。"""
        self.res = None
        self.b_poni.setEnabled(False)
        self.b_copy.setEnabled(False)
        self.b_check.setEnabled(False)

    # ---------- 事件 ----------
    def open_img(self):
        start = self.st.value("last_dir", "") or ""
        p, _ = QFileDialog.getOpenFileName(
            self, "选择衍射图", start,
            "衍射图 (*.edf *.cbf *.h5 *.hdf5 *.tif *.tiff *.mar* *.png *.jpg);;所有 (*.*)")
        if p:
            self._load(p)

    def _load(self, p):
        try:
            self.img = load_any(p)
        except Exception as e:
            QMessageBox.warning(self, "打开失败", str(e))
            return
        self.st.setValue("last_dir", os.path.dirname(p))
        self.poni_path = os.path.splitext(p)[0] + "_fit.poni"
        self.lbl_file.setText(os.path.basename(p))
        self.img_raw = None           # 换图即清校正态（F6：不选 spline 不受影响）
        self.spline = None
        self.b_sp.setEnabled(True)
        self._sp_refresh()
        self.show_img()
        self.b_run.setEnabled(True)
        self.b_poni.setEnabled(False)
        self.res = None
        meta = sniff_meta(p)
        if "px" in meta:
            self.e_px.setText(f"{meta['px']:g}")
        if "lam" in meta:
            self.e_lam.setText(f"{meta['lam']:g}")
        extra = ""
        if meta["src"]:
            extra = (f"\n\n文件头自动读取（{meta['src']}）:\n"
                     f"  像素 = {meta.get('px', '—')} µm"
                     f"{'（已填入）' if 'px' in meta else ''}")
            if "lam" in meta:
                extra += f"\n  波长 = {meta['lam']:g} Å（已填入，清空=恢复盲解）"
            if "D_mm" in meta:
                extra += f"\n  文件头自述距离 D = {meta['D_mm']:.1f} mm（求解不用，仅对照）"
            if "beam" in meta:
                extra += (f"\n  文件头自述束心 = ({meta['beam'][0]:.1f},"
                          f" {meta['beam'][1]:.1f}) px（求解不用，仅对照）")
        self.txt.setPlainText(f"图像: {os.path.basename(p)}\n"
                              f"尺寸: {self.img.shape[1]}×{self.img.shape[0]} px\n"
                              f"动态范围: {self.img.min():.0f} ~ {self.img.max():.0f}"
                              f"{extra}\n\n提示: 相鉴定/中心不需要像素与波长；\n"
                              f"物理 D 靠像素，小角数据想定 D 还需 λ。")
        self.status.setText("已加载，按 ▶ 运行求解")

    def show_img(self, res=None):
        self.ax.clear()
        im = np.clip(self.img, *np.percentile(self.img, [2, 99.5]))
        self.ax.imshow(im, cmap="gray", origin="upper")
        self.ax.set_title("校正图（2–99.5% 显示）" if self.spline
                          else "原始图（2–99.5% 显示）")
        if res is not None:
            H, W = self.img.shape
            cx, cy, D_px, lam = (res["cx"], res["cy"],
                                 res["D_px"], res["lam"])
            for q in phase_qs(res["ph"]):
                r = _r_model(q, lam, D_px)
                if 3 < r < 1.35 * max(H, W):
                    self.ax.add_patch(Circle((cx, cy), r, fill=False,
                                             ec="#009E73", lw=1.0, alpha=0.75))
            if res.get("native"):
                self.ax.add_patch(Circle((cx, cy), 12, fill=False,
                                         ec="yellow", lw=1.4, alpha=0.9))
            self.ax.plot([cx], [cy], "r+", ms=16, mew=2)
            tag = "校正图" if self.spline else "原始图"
            self.ax.set_title(f"{res['ph']} 环系预测（绿）· 束心（红十字）· {tag}")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.tight_layout()
        self.cv.draw_idle()

    def run_solve(self):
        if self.worker is not None and self.worker.isRunning():
            return
        px = float(self.e_px.text() or 172.0)
        lam_txt = self.e_lam.text().strip()
        self.worker = SolveWorker(self.img, float(lam_txt) if lam_txt else None, px)
        self.worker.done.connect(self.on_done)
        self.worker.fail.connect(self.on_fail)
        self.b_run.setEnabled(False); self.pb.setVisible(True)
        self.status.setText("求解中——画布谱 → 相仲裁 → 2D 精化…")
        self.worker.start()

    def on_fail(self, tb):
        self.pb.setVisible(False); self.b_run.setEnabled(True)
        self.status.setText("失败")
        QMessageBox.critical(self, "求解异常", tb)

    def on_done(self, res):
        self.pb.setVisible(False); self.b_run.setEnabled(True)
        if res is None:
            self.status.setText("解不出（环太少/太弱/相库外）")
            self.txt.append("\n— 解不出：环太少/太弱/相库外，诚实拒收。")
            return
        self.res = res
        self.b_poni.setEnabled(True)
        self.b_copy.setEnabled(True)
        self.b_check.setEnabled(True)
        self.st.setValue("px", self.e_px.text())
        self.st.setValue("lam", self.e_lam.text())
        px_um = float(self.e_px.text() or 172.0)
        D_mm = res["D_px"] * px_um * 1e-3
        sD = res.get("sD_mm", 0.0)
        d_line = (f"距离 D: {D_mm:.2f} ± {sD:.2f} mm（环间一致性 1σ）"
                  if sD > 0 else f"距离 D: {D_mm:.2f} mm  ± 未估（环不足）")
        lines = [
            f"物相: {res['ph']}",
            f"波长 λ̂: {res['lam']:.4f} Å" + ("" if not res["native"] else "（联合解）"),
            d_line,
            f"束心: ({res['cx']:.1f}, {res['cy']:.1f}) px（原生）",
            f"匹配环数 m={res['sol'][3]}  rms={res['sol'][4]:.2f}px  ps={res['sol'][6]:.3f}",
            f"2D 精化: {'✓ ' + str(res['n2d']) + ' 环（中心已用）' if res['native'] else '环不完整→跳过（部分弧 Kasa 易拉偏）'}",
        ]
        q_rms = res.get("ring_rms", 0.0)
        if q_rms > 0:
            lines.append(f"几何自洽: 环拟合残差 rms {q_rms:.2f}px"
                         f"（{res.get('ring_n', 0)} 环；中心取自 "
                         f"{res.get('center_src', '?')}）")
        if res.get("tilt_on") and res.get("tf"):
            tf = res["tf"]
            s1 = res.get("s_r1", 0.0) * 1e3
            s2 = res.get("s_r2", 0.0) * 1e3
            tilt_mag = np.hypot(tf["rot1"], tf["rot2"]) * 180 / np.pi
            err = (f" ±({s1:.1f},{s2:.1f})" if (s1 or s2) else "")
            lines.append(f"倾角: rot1={tf['rot1']*1e3:+.1f}{err} rot2={tf['rot2']*1e3:+.1f} mrad"
                         f"（合倾角 {tilt_mag:.2f}°；σ 为统计 1σ，系统差另计）")
            lines.append(f"        残差 {tf['rms5']*1e3:.2f}mrad·改善 {(tf.get('imp') or 0)*100:.0f}%，"
                         f"{tf['npts']} 控制点（部分弧可用）——poni/自检已含")
        elif res.get("native") and res.get("tf"):
            tf = res["tf"]
            s1 = res.get("s_r1", 0.0) * 1e3
            s2 = res.get("s_r2", 0.0) * 1e3
            lines.append(f"倾角（诊断，未并入 poni）: rot1={tf['rot1']*1e3:+.1f}±{s1:.1f} "
                         f"rot2={tf['rot2']*1e3:+.1f}±{s2:.1f} mrad"
                         f"（改善 {(tf.get('imp') or 0)*100:.0f}%<5%——小角数据倾角"
                         f"不可辨识，硬加反而掉精度）")
        elif res.get("native"):
            lines.append("倾角: 跳过（T7 已给亚像素中心，倾角改善不足 5%——小角数据倾角"
                         "不可辨识，硬加反而掉精度；P2M 实证 0.010° vs 0.138°）")
        else:
            lines.append("倾角: 未拟合（rot=0；峰位 0.1° 级误差风险）")
        if not self.e_lam.text().strip():
            lines.append("⚠ λ 未给定：小角域 λ–D 简并，D 与 λ̂ 仅乘积可信")
        self.txt.setPlainText("\n".join(lines))
        self.status.setText(f"完成 · {res['ph']}")
        self.show_img(res)
        self.plot_radial(res)

    def plot_radial(self, res):
        """右栏径向谱：画布径向谱 + 检测峰 + 相预测环（Okabe-Ito 色盲安全）。"""
        try:
            import theme
            s = res["s"]
            cx, cy = res["cx"] / s, res["cy"] / s
            # R3：fallback 采用时画布与默认档不同——必须用求解时那张
            # （res["canvas"]），否则径向谱与解错位。
            canvas = res.get("canvas")
            if canvas is None:
                canvas, _, _, _ = to_canvas(self.img)
            peaks, prof, bp, good = ring_radii(canvas, cx, cy)
            self.ax2.clear()
            r = np.arange(len(prof)) * bp
            self.ax2.plot(r, prof, lw=1.0, color=theme.C_DATA, label="径向谱")
            if peaks:
                self.ax2.plot([p[0] for p in peaks], [p[1] for p in peaks],
                              "v", ms=3, color=theme.C_PEAK, label="检测峰")
            for q in np.asarray(phase_qs(res["ph"]), float):
                rr = _r_model(q, res["lam"], res["D_px"])
                if 3 < rr < r.max():
                    self.ax2.axvline(rr, color=theme.C_PRED, lw=0.8, alpha=0.6)
            self.ax2.set_xlabel("r (px)", fontsize=8)
            self.ax2.set_ylabel("I (a.u.)", fontsize=8)
            self.ax2.set_title(f"径向谱 · {res['ph']} 预测环（绿线）", fontsize=9)
            self.ax2.grid(alpha=0.25, lw=0.5)
            self.ax2.tick_params(labelsize=7)
            self.ax2.legend(fontsize=6, frameon=False, loc="upper right")
            theme.restyle_axes(self.ax2, self.theme_name)
            self.fig2.tight_layout()
            self.cv2.draw_idle()
        except Exception:
            pass

    def run_check(self):
        if self.res is None:
            return
        if getattr(self, "checker", None) is not None and self.checker.isRunning():
            return
        self.b_check.setEnabled(False); self.pb.setVisible(True)
        self.status.setText("标样自检中——积分 16000 点 → 找峰 → 对答案卡…")
        self.checker = CheckWorker(self.img, self.res,
                                   float(self.e_px.text() or 172.0))
        self.checker.done.connect(self.on_check_done)
        self.checker.fail.connect(self.on_check_fail)
        self.checker.start()

    def on_check_fail(self, tb):
        self.pb.setVisible(False); self.b_check.setEnabled(True)
        self.status.setText("自检失败")
        QMessageBox.critical(self, "自检异常", tb)

    def on_check_done(self, c):
        self.pb.setVisible(False); self.b_check.setEnabled(True)
        head = (f"\n—— 标样自检（{self.res['ph']} 答案卡，"
                f"{c['n']} 环参评）——")
        self.txt.append(head)
        for ln in c["lines"]:
            self.txt.append(ln)
        self.txt.append("  " + c["verdict"])
        self.status.setText(f"自检：{c['verdict']}")

    def export_poni(self):
        if self.res is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "保存 poni",
                                           self.poni_path or "fit.poni",
                                           "poni (*.poni)")
        if not p:
            return
        px_um = float(self.e_px.text() or 172.0)
        ai = hybrid_ai(self.res, px_um)  # 与自检同源（终极混合五参）
        if self.spline:
            # F4：poni 带 SplineFile，下游 pyFAI 对原始图按同一套畸变处理
            # （不写则 poni 只对校正图有效，用户拿原始数据用会错）
            set_splinefile(ai, self.spline)
        ai.save(p)
        self.status.setText(f"已导出 {p}" + ("（含 SplineFile）" if self.spline else ""))


def main():
    app = QApplication(sys.argv)
    win = Win()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


