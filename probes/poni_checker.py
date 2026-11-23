# -*- coding: utf-8 -*-
"""poni 验证器 v2（独立小程序）

判据升级：不再用"环带均值"，而是**用 poni 真正积分一次**，看衍射环在一维谱里
有多锐（峰宽 FWHM）。峰越窄 = 几何越准。倾角、视差都由 pyFAI 处理。

用法：
    python poni_checker.py                       # 图形界面
    python poni_checker.py --img X --a A.poni --b B.poni --phase AgBh --run
"""
import json
import os
import re
import sys
import tempfile
import traceback

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QApplication, QComboBox, QFileDialog, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPushButton, QTextEdit, QVBoxLayout,
                             QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavBar
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fabio                                   # noqa: E402
import pyFAI                                   # noqa: E402
from solver_core import auto_offset            # noqa: E402
from torchfai.phases import PHASES, phase_qs   # noqa: E402

PHASE_NAMES = list(PHASES.keys())
VERSION = "2.1"


# ---------------------------------------------------------------- poni 处理
def parse_poni(p):
    kv = {}
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"^(\w+):\s*(.+)$", line.strip())
        if m:
            kv[m.group(1)] = m.group(2)
        mc = re.match(r"^#\s*Calibrant:\s*(\S+)", line.strip())
        if mc:
            kv["_cal"] = mc.group(1)
    cfg = json.loads(kv["Detector_config"])
    px = float(cfg.get("pixel1", cfg.get("pixel", 1e-4)))
    return dict(px=px, dist=float(kv["Distance"]),
                p1=float(kv["Poni1"]), p2=float(kv["Poni2"]),
                lam=float(kv["Wavelength"]) * 1e10 if "Wavelength" in kv else None,
                cal=kv.get("_cal"))


def _tmp_poni(src, swap=False):
    """写一份 ASCII 路径、可选交换 Poni1/Poni2 的临时 poni（pyFAI 需要 utf-8 头）。"""
    lines = []
    for line in open(src, encoding="utf-8", errors="replace"):
        s = line.rstrip("\n")
        if s.startswith("# Image:") or s.startswith("#Image:"):
            continue
        lines.append(s)
    if swap:
        out = []
        for s in lines:
            if s.startswith("Poni1:"):
                out.append("__SWAP1__")
            elif s.startswith("Poni2:"):
                out.append("Poni1:" + s.split(":", 1)[1])
            else:
                out.append(s)
        v1 = [s for s in lines if s.startswith("Poni1:")][0].split(":", 1)[1]
        lines = ["Poni2:" + v1 if s == "__SWAP1__" else s for s in out]
    fd, path = tempfile.mkstemp(suffix=".poni")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def integrate(img, poni_path, swap=False, npt=2000):
    tmp = _tmp_poni(poni_path, swap=swap)
    try:
        ai = pyFAI.load(tmp)
        q, I = ai.integrate1d(img, npt, unit="q_A^-1", method="bbox,splitpixel")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return q, I, ai


def ring_metrics(q, I, qs, frac=0.12):
    """对每条理论环，在 q 附近找峰，返回 (峰宽列表, 对比度列表, 命中数)。"""
    fwhm, contrast, hits = [], [], 0
    for q0 in qs:
        sel = (q > q0 * (1 - frac)) & (q < q0 * (1 + frac))
        if sel.sum() < 8:
            continue
        qs_, Is = q[sel], I[sel]
        base = float(np.median(Is))
        k = int(np.argmax(Is))
        peak = float(Is[k])
        if peak <= 1.3 * max(base, 1e-9):
            fwhm.append(np.nan)
            contrast.append(float(peak / max(base, 1e-9)))
            continue
        half = base + (peak - base) / 2
        l = k
        while l > 0 and Is[l] > half:
            l -= 1
        r = k
        while r < len(Is) - 1 and Is[r] > half:
            r += 1
        fwhm.append(float(qs_[r] - qs_[l]))
        contrast.append(float(peak / max(base, 1e-9)))
        hits += 1
    return fwhm, contrast, hits


def analyse(img, poni_path, phase):
    """两种轴映射都试，取峰宽更窄的那个。"""
    gt = parse_poni(poni_path)
    qs = np.asarray(phase_qs(phase), float)
    best = None
    for swap in (False, True):
        try:
            q, I, ai = integrate(img, poni_path, swap=swap)
        except Exception:
            continue
        fw, ct, hits = ring_metrics(q, I, qs)
        fin = [x for x in fw if np.isfinite(x)]
        if not fin:
            continue
        med = float(np.median(fin))
        if best is None or med < best["fwhm"]:
            best = dict(fwhm=med, contrast=float(np.nanmedian(ct)),
                        hits=hits, ntot=len(qs), swap=swap,
                        q_peak=float(q[np.argmax(I)]), ai=ai)
    if best is None:
        raise RuntimeError("两种轴映射都积不出可用谱")
    # 圆心：必须用 pyFAI 的投影值（getFit2D），不能直接读 Poni1/Poni2——
    # 有倾角时两者不等，直接读会得出"圆心在画面外"的错误结论。
    # 注意：必须用 best 里那份 ai，不能用循环里泄漏出来的（可能是 swap 版）
    try:
        f2d = best["ai"].getFit2D()
        cx, cy = float(f2d["centerX"]), float(f2d["centerY"])
        tilt = float(np.degrees(f2d.get("tilt", 0) or 0))
    except Exception:
        cx = (gt["p1"] if best["swap"] else gt["p2"]) / gt["px"] - 0.5
        cy = (gt["p2"] if best["swap"] else gt["p1"]) / gt["px"] - 0.5
        tilt = float("nan")
    best.update(cx=cx, cy=cy, tilt=tilt, D=gt["dist"], px=gt["px"], gt=gt)
    return best


def ring_radii_px(gt, qs):
    lam = gt["lam"]
    theta = np.arcsin(np.clip(lam * qs / (4 * np.pi), 0, 0.999))
    return gt["dist"] * np.tan(2 * theta) / gt["px"]


# ---------------------------------------------------------------- 界面
class Win(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"poni 验证器 v{VERSION} · 环锐度检查")
        self.resize(1180, 760)
        self.img = None
        self.img_path = None
        self.pa = self.pb = None
        self.ra = self.rb = None

        root = QWidget()
        self.setCentralWidget(root)
        h = QHBoxLayout(root)

        left = QWidget()
        left.setMaximumWidth(340)
        lv = QVBoxLayout(left)

        g1 = QGroupBox("1. 图像")
        l1 = QVBoxLayout(g1)
        self.b_img = QPushButton("打开图像…")
        self.b_img.clicked.connect(self.open_img)
        self.l_img = QLabel("（未载入）")
        self.l_img.setWordWrap(True)
        l1.addWidget(self.b_img)
        l1.addWidget(self.l_img)
        lv.addWidget(g1)

        g2 = QGroupBox("2. poni")
        l2 = QVBoxLayout(g2)
        self.b_a = QPushButton("载入 poni A…")
        self.b_a.clicked.connect(lambda: self.open_poni("a"))
        self.l_a = QLabel("A: （未载入）")
        self.l_a.setWordWrap(True)
        self.b_b = QPushButton("载入 poni B（对比）…")
        self.b_b.clicked.connect(lambda: self.open_poni("b"))
        self.l_b = QLabel("B: （未载入）")
        self.l_b.setWordWrap(True)
        for w in (self.b_a, self.l_a, self.b_b, self.l_b):
            l2.addWidget(w)
        lv.addWidget(g2)

        g3 = QGroupBox("3. 参数")
        l3 = QVBoxLayout(g3)
        row = QHBoxLayout()
        row.addWidget(QLabel("标样相"))
        self.cb_ph = QComboBox()
        self.cb_ph.addItems(["（从 poni 读）"] + PHASE_NAMES)
        row.addWidget(self.cb_ph)
        l3.addLayout(row)
        lv.addWidget(g3)

        self.b_run = QPushButton("▶  检查环锐度")
        self.b_run.clicked.connect(self.run_check)
        lv.addWidget(self.b_run)
        self.b_png = QPushButton("导出图…")
        self.b_png.clicked.connect(self.export_png)
        lv.addWidget(self.b_png)

        g4 = QGroupBox("结果")
        l4 = QVBoxLayout(g4)
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        l4.addWidget(self.txt)
        lv.addWidget(g4, 1)
        h.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.fig = Figure(figsize=(6, 6))
        self.cv = FigureCanvas(self.fig)
        rv.addWidget(NavBar(self.cv, right))
        rv.addWidget(self.cv, 1)
        self.status = QLabel("判据：用 poni 积分后，衍射环的峰宽（越窄越好）")
        rv.addWidget(self.status)
        h.addWidget(right, 1)

    def open_img(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "打开衍射图", "",
            "图像 (*.tif *.tiff *.cbf *.edf *.npy);;所有文件 (*)")
        if not p:
            return
        try:
            d = np.load(p) if p.lower().endswith(".npy") else fabio.open(p).data
            self.img = auto_offset(np.asarray(d).astype(np.float32))
            self.img_path = p
            self.l_img.setText(f"{os.path.basename(p)}\n"
                               f"{self.img.shape[1]}×{self.img.shape[0]}")
        except Exception as e:
            QMessageBox.warning(self, "打开失败", str(e))

    def open_poni(self, which):
        p, _ = QFileDialog.getOpenFileName(self, "打开 poni", "",
                                           "poni (*.poni);;所有文件 (*)")
        if not p:
            return
        try:
            gt = parse_poni(p)
        except Exception as e:
            QMessageBox.warning(self, "解析失败", str(e))
            return
        txt = (f"D={gt['dist']*1000:.1f} mm, px={gt['px']*1e6:.1f} µm"
               + (f", 标样={gt['cal']}" if gt.get("cal") else ""))
        if which == "a":
            self.pa, self.pa_path = gt, p
            self.l_a.setText(f"A: {os.path.basename(p)}\n{txt}")
            if gt.get("cal") in PHASE_NAMES:
                self.cb_ph.setCurrentText(gt["cal"])
        else:
            self.pb, self.pb_path = gt, p
            self.l_b.setText(f"B: {os.path.basename(p)}\n{txt}")

    def run_check(self):
        if self.img is None or self.pa is None:
            QMessageBox.information(self, "提示", "请先载入图像和 poni A")
            return
        ph = self.cb_ph.currentText()
        if ph.startswith("（"):
            ph = self.pa.get("cal")
        if ph not in PHASE_NAMES:
            QMessageBox.information(self, "提示", "请选择标样相（如 AgBh / LaB6）")
            return
        try:
            self.ra = analyse(self.img, self.pa_path, ph)
            self.rb = (analyse(self.img, self.pb_path, ph)
                       if self.pb is not None else None)
        except Exception:
            QMessageBox.warning(self, "检查失败", traceback.format_exc(limit=3))
            return
        self.report(ph)
        self.draw(ph)

    @staticmethod
    def grade(r):
        f = r["fwhm"]
        return ("✅ 很锐" if f < 3e-3 else "✓ 可用" if f < 6e-3 else
                "⚠ 偏宽" if f < 1.2e-2 else "❌ 很宽")

    def report(self, ph):
        a = self.ra
        lines = [f"标样相：{ph}",
                 f"A  {self.grade(a)}",
                 f"   峰宽 FWHM = {a['fwhm']:.5f} Å⁻¹   对比度 {a['contrast']:.1f}×",
                 f"   命中环 {a['hits']}/{a['ntot']}"
                 f"   轴映射 {'Poni1↔Poni2' if a['swap'] else '直接'}",
                 f"   圆心 ({a['cx']:.1f}, {a['cy']:.1f})  D={a['D']*1000:.2f} mm"
                 f"  tilt={a['tilt']:.2f}°"]
        if self.rb is not None:
            b = self.rb
            lines += ["", f"B  {self.grade(b)}",
                      f"   峰宽 FWHM = {b['fwhm']:.5f} Å⁻¹   对比度 {b['contrast']:.1f}×",
                      f"   命中环 {b['hits']}/{b['ntot']}"
                      f"   轴映射 {'Poni1↔Poni2' if b['swap'] else '直接'}",
                      f"   圆心 ({b['cx']:.1f}, {b['cy']:.1f})  D={b['D']*1000:.2f} mm"
                      f"  tilt={b['tilt']:.2f}°"]
            better = "A" if a["fwhm"] < b["fwhm"] else "B"
            ratio = max(a["fwhm"], b["fwhm"]) / max(min(a["fwhm"], b["fwhm"]), 1e-9)
            lines += ["", f"→ {better} 更准（峰宽比 {ratio:.2f}×）"]
        self.txt.setPlainText("\n".join(lines))
        self.status.setText(f"{ph}  A: FWHM {a['fwhm']:.5f}")

    def draw(self, ph):
        """用 pyFAI 的真实几何画环：对 2θ 场做等值线，天然包含倾角与视差。"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        lo, hi = np.percentile(self.img, [1, 99.8])
        v = np.clip((self.img - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.45
        ax.imshow(v, cmap="gray", origin="upper")
        qs = np.asarray(phase_qs(ph), float)
        for tag, r in (("a", self.ra), ("b", self.rb)):
            if r is None:
                continue
            col = "#39a275" if tag == "a" else "#f6ad55"
            ai = r.get("ai")
            try:
                tth = ai.tth            # 2θ 场（弧度），含倾角/视差
                lam = r["gt"]["lam"] * 1e-10
                tgt = 2 * np.arcsin(np.clip(lam * qs / (4 * np.pi), 0, 0.999))
                inrange = tgt[(tgt > tth.min() * 1.02) & (tgt < tth.max() * 0.98)]
                if inrange.size:
                    ax.contour(tth, levels=list(inrange), colors=[col],
                               linewidths=0.8, alpha=0.9)
            except Exception:
                # 兜底：退回正圆
                th = np.linspace(0, 2 * np.pi, 900)
                gt = r.get("gt") or self.pa
                for r0 in ring_radii_px(gt, qs):
                    if 20 < r0 < 0.5 * min(self.img.shape):
                        ax.plot(r["cx"] + r0 * np.cos(th),
                                r["cy"] + r0 * np.sin(th),
                                lw=0.8, color=col, alpha=0.9)
            ax.plot([r["cx"]], [r["cy"]], marker="+", ms=12, color=col, mew=1.6)
        ax.set_xlim(0, self.img.shape[1])
        ax.set_ylim(self.img.shape[0], 0)
        ax.set_title(f"{os.path.basename(self.img_path or '')} — {ph}"
                     + ("   (green=A, orange=B)" if self.rb else ""), fontsize=9)
        self.cv.draw_idle()

    def export_png(self):
        if self.ra is None:
            QMessageBox.information(self, "提示", "先做一次检查")
            return
        p, _ = QFileDialog.getSaveFileName(self, "保存图", "poni_check.png",
                                           "PNG (*.png)")
        if p:
            self.fig.savefig(p, dpi=220)
            self.status.setText(f"已保存 {os.path.basename(p)}")


def _argval(flag):
    a = sys.argv
    if flag in a:
        i = a.index(flag)
        if i + 1 < len(a):
            return a[i + 1]
    return None


def main():
    app = QApplication(sys.argv)
    w = Win()
    w.show()

    img_p, a_p, b_p = _argval("--img"), _argval("--a"), _argval("--b")
    ph = _argval("--phase")
    if img_p:
        try:
            w.img = auto_offset(np.asarray(fabio.open(img_p).data).astype(np.float32))
            w.img_path = img_p
            w.l_img.setText(f"{os.path.basename(img_p)}\n"
                            f"{w.img.shape[1]}×{w.img.shape[0]}")
        except Exception as e:
            print("图像载入失败:", e)
    for path, which in ((a_p, "a"), (b_p, "b")):
        if not path:
            continue
        try:
            gt = parse_poni(path)
        except Exception as e:
            print("poni 解析失败:", e)
            continue
        if which == "a":
            w.pa, w.pa_path = gt, path
            w.l_a.setText(f"A: {os.path.basename(path)}")
        else:
            w.pb, w.pb_path = gt, path
            w.l_b.setText(f"B: {os.path.basename(path)}")
    if ph:
        w.cb_ph.setCurrentText(ph)
    if "--run" in sys.argv and w.img is not None and w.pa is not None:
        QTimer.singleShot(400, w.run_check)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
