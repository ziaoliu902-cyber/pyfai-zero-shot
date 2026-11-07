# -*- coding: utf-8 -*-
"""batch_gui.py：批量模式（2026-11-07 A2）。

文件夹/多选文件 → 逐帧盲解（SolveWorker.run 同步复用；环不足=无人值守
自动爬档，与 EscalationFlow 同策略无对话框）→ 表格（物相/D/m/rms/ps/σ_D/
状态）→ CSV / 全部 .poni 一键导出 → ≥5 帧成功画 D-漂移曲线（B9 集成点）。
每帧像素可自动嗅探（sniff_meta），嗅不到用全局值兜底。
启动：主窗口「📦 批量…」按钮（gui_solver.Win.open_batch）。
"""
import os
import csv

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QProgressBar, QFileDialog, QCheckBox,
    QComboBox, QMessageBox, QHeaderView)

from gui_solver import load_any, sniff_meta, SolveWorker, UncWorker, hybrid_ai
from bootstrap_uncertainty import d_bootstrap

IMG_EXT = (".edf", ".cbf", ".h5", ".hdf5", ".tif", ".tiff",
           ".mar2300", ".mar3450", ".mar165", ".png", ".jpg")


class BatchWorker(QThread):
    """逐帧盲解队列;emit row(i,dict)/prog(i,n)/done_all()。"""
    row = pyqtSignal(int, dict)
    prog = pyqtSignal(int, int)
    done_all = pyqtSignal()

    def __init__(self, files, px_um, lam, want_sigma=True, sens=0):
        super().__init__()
        self.files = files
        self.px_um = px_um
        self.lam = lam
        self.want_sigma = want_sigma
        self.sens = sens
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    # ---- 单帧:SolveWorker 同步 + 无人值守爬档(环不足才升,同主窗口策略) ----
    def _solve_one(self, img, px_um, lam):
        import solver_core as _sc
        from_ = _sc._RING_SENS
        _sc.set_ring_sensitivity(from_)
        out = {}
        while True:
            w = SolveWorker(img, lam, px_um)
            w.done.connect(lambda r: out.__setitem__("r", r))
            w.fail.connect(lambda tb: out.__setitem__("tb", tb))
            w.run()                        # 同步执行(不进事件循环)
            if "tb" in out:
                raise RuntimeError(out["tb"].splitlines()[-1])
            if out.get("r") is not None:
                return out["r"], _sc._RING_SENS != from_
            if _sc._RING_SENS >= 2:        # 爬完仍无解 → 诚实拒收
                _sc.set_ring_sensitivity(from_)
                return None, False
            _sc.set_ring_sensitivity(_sc._RING_SENS + 1)

    def run(self):
        n = len(self.files)
        for i, f in enumerate(self.files):
            if self.stop_flag:
                break
            d = {"file": f, "ph": "", "D_mm": "", "m": "", "rms": "",
                 "ps": "", "sD": "", "status": ""}
            try:
                px_um = self.px_um
                if self._auto_px:
                    m = sniff_meta(f)
                    if "px" in m:
                        px_um = float(m["px"])
                img = load_any(f)
                res, climbed = self._solve_one(img, px_um, self.lam)
                d["px"] = px_um
                if res is None:
                    d["status"] = "解不出（环太少/太弱/相库外）"
                else:
                    d["ph"] = res["ph"]
                    d["D_mm"] = round(res["D_px"] * px_um * 1e-3, 2)
                    d["m"] = res["sol"][3]
                    d["rms"] = round(res["sol"][4], 2)
                    d["ps"] = round(res["sol"][6], 3)
                    d["status"] = ("✓(环不足自动调档)" if climbed else "✓")
                    d["res"] = res
                    if self.want_sigma:
                        from torchfai.phases import phase_qs
                        bs = d_bootstrap(
                            img, res["cx"], res["cy"], res["D_px"],
                            phase_qs(res["ph"]), res["lam"], px_um * 1e-6)
                        if isinstance(bs, tuple) and len(bs) == 4:
                            d["sD"] = round(bs[0], 2)
            except Exception as e:
                d["status"] = f"异常: {e}"
            self.row.emit(i, d)
            self.prog.emit(i + 1, n)
        self.done_all.emit()


class BatchDialog(QDialog):
    """批量主窗:参数行 + 表格 + 导出/漂移曲线。"""

    def __init__(self, parent=None, px_um=172.0, lam=""):
        super().__init__(parent)
        self.setWindowTitle("批量盲解 · DiffBlindWizard")
        self.resize(980, 560)
        self.worker = None
        self.rows = []
        self.files = []

        lay = QVBoxLayout()
        top = QHBoxLayout()
        self.b_dir = QPushButton("选文件夹…")
        self.b_dir.clicked.connect(self.pick_dir)
        self.b_files = QPushButton("选文件…")
        self.b_files.clicked.connect(self.pick_files)
        self.lab_files = QLabel("（未选择——0 帧）")
        top.addWidget(self.b_dir); top.addWidget(self.b_files)
        top.addWidget(self.lab_files, 1)
        lay.addLayout(top)

        par = QHBoxLayout()
        par.addWidget(QLabel("像素 µm:"))
        self.e_px = QLineEdit(f"{px_um:g}"); self.e_px.setMaximumWidth(80)
        self.cb_pxauto = QCheckBox("自动嗅探"); self.cb_pxauto.setChecked(True)
        par.addWidget(self.e_px); par.addWidget(self.cb_pxauto)
        par.addWidget(QLabel("波长 Å:"))
        self.e_lam = QLineEdit(lam or ""); self.e_lam.setMaximumWidth(90)
        self.e_lam.setPlaceholderText("空=联立解")
        par.addWidget(self.e_lam)
        par.addWidget(QLabel("灵敏度:"))
        self.cb_sens = QComboBox(); self.cb_sens.addItems(["标准", "高", "极高"])
        par.addWidget(self.cb_sens)
        self.cb_sigma = QCheckBox("σ(D)"); self.cb_sigma.setChecked(True)
        self.cb_sigma.setToolTip("每帧环级 bootstrap(约 +1~3s/帧)")
        par.addWidget(self.cb_sigma)
        self.b_run = QPushButton("▶ 开始")
        self.b_run.setProperty("primary", True)
        self.b_run.clicked.connect(self.run_batch)
        self.b_stop = QPushButton("■ 停止"); self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(self.stop_batch)
        par.addWidget(self.b_run); par.addWidget(self.b_stop)
        lay.addLayout(par)

        self.pb = QProgressBar(); self.pb.setVisible(False)
        lay.addWidget(self.pb)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["文件", "物相", "D (mm)", "m", "rms(px)", "ps", "σ_D", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 1)

        bot = QHBoxLayout()
        self.b_csv = QPushButton("导出 CSV…"); self.b_csv.setEnabled(False)
        self.b_csv.clicked.connect(self.export_csv)
        self.b_poni = QPushButton("导出全部 .poni"); self.b_poni.setEnabled(False)
        self.b_poni.clicked.connect(self.export_poni_all)
        self.b_drift = QPushButton("📈 D 漂移曲线"); self.b_drift.setEnabled(False)
        self.b_drift.clicked.connect(self.show_drift)
        self.b_close = QPushButton("关闭"); self.b_close.clicked.connect(self.accept)
        for b in (self.b_csv, self.b_poni, self.b_drift, self.b_close):
            bot.addWidget(b)
        lay.addLayout(bot)
        self.setLayout(lay)

    # ---------- 选源 ----------
    def _set_files(self, fs):
        self.files = sorted(fs)
        self.lab_files.setText(f"（{len(self.files)} 帧待解）")
        self.table.setRowCount(0)
        self.rows = []

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选衍射图文件夹")
        if d:
            self._set_files([os.path.join(d, x) for x in os.listdir(d)
                             if x.lower().endswith(IMG_EXT)])

    def pick_files(self):
        fs, _ = QFileDialog.getOpenFileNames(
            self, "选衍射图", "",
            "衍射图 (*.edf *.cbf *.h5 *.hdf5 *.tif *.tiff *.mar* *.png *.jpg);;所有 (*.*)")
        if fs:
            self._set_files(fs)

    # ---------- 跑 ----------
    def run_batch(self):
        if not self.files:
            QMessageBox.information(self, "批量", "先选文件夹或文件。")
            return
        px = float(self.e_px.text() or 172.0)
        lam_t = self.e_lam.text().strip()
        import solver_core as _sc
        _sc.set_ring_sensitivity(self.cb_sens.currentIndex())
        self.worker = BatchWorker(self.files, px, float(lam_t) if lam_t else None,
                                  want_sigma=self.cb_sigma.isChecked(),
                                  sens=self.cb_sens.currentIndex())
        self.worker._auto_px = self.cb_pxauto.isChecked()
        self.table.setRowCount(len(self.files))
        self.rows = [None] * len(self.files)
        self.worker.row.connect(self.on_row)
        self.worker.prog.connect(self.on_prog)
        self.worker.done_all.connect(self.on_done_all)
        self.b_run.setEnabled(False); self.b_stop.setEnabled(True)
        self.pb.setRange(0, len(self.files)); self.pb.setValue(0)
        self.pb.setVisible(True)
        self.worker.start()

    def stop_batch(self):
        if self.worker is not None:
            self.worker.stop()

    def on_row(self, i, d):
        self.rows[i] = d
        vals = [os.path.basename(d["file"]), d["ph"], d["D_mm"], d["m"],
                d["rms"], d["ps"], d["sD"], d["status"]]
        for c, v in enumerate(vals):
            it = QTableWidgetItem(str(v))
            if c == 7:
                ok = str(v).startswith("✓")
                it.setForeground(Qt.darkGreen if ok else Qt.darkRed)
            self.table.setItem(i, c, it)

    def on_prog(self, i, n):
        self.pb.setValue(i)

    def on_done_all(self):
        self.b_run.setEnabled(True); self.b_stop.setEnabled(False)
        self.pb.setVisible(False)
        ok = [r for r in self.rows if r and r.get("res")]
        self.b_csv.setEnabled(True)
        self.b_poni.setEnabled(bool(ok))
        self.b_drift.setEnabled(len(ok) >= 5)
        QMessageBox.information(
            self, "批量完成",
            f"{len(ok)}/{len(self.rows)} 帧解出。"
            + ("漂移曲线可用（≥5 帧）。" if len(ok) >= 5 else ""))

    # ---------- 导出 ----------
    def export_csv(self):
        p, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "批量结果.csv",
                                           "CSV (*.csv)")
        if not p:
            return
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["文件", "物相", "D_mm", "m", "rms_px", "ps",
                         "sigma_D_mm", "状态", "像素_um"])
            for r in self.rows:
                if r:
                    wr.writerow([r["file"], r["ph"], r["D_mm"], r["m"],
                                 r["rms"], r["ps"], r["sD"], r["status"],
                                 r.get("px", "")])
        QMessageBox.information(self, "CSV", f"已导出 {p}")

    def export_poni_all(self):
        n = 0
        for r in self.rows:
            if r and r.get("res"):
                try:
                    ai = hybrid_ai(r["res"], r["px"])
                    ai.save(os.path.splitext(r["file"])[0] + "_fit.poni")
                    n += 1
                except Exception:
                    pass
        QMessageBox.information(self, "poni", f"已导出 {n} 个 <图名>_fit.poni")

    # ---------- 漂移曲线(B9 集成点) ----------
    def show_drift(self):
        import numpy as np
        import matplotlib
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        pts = [(i, r["D_mm"], r["sD"] or 0.0)
               for i, r in enumerate(self.rows) if r and r.get("res")]
        if len(pts) < 2:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        dlg = QDialog(self)
        dlg.setWindowTitle("D 漂移曲线（帧序 vs 距离）")
        dlg.resize(720, 420)
        lay = QVBoxLayout(dlg)
        fig = Figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=4, lw=1,
                    ecolor="#888", capsize=2)
        med = float(np.median(ys))
        ax.axhline(med, ls="--", lw=0.8, color="#C44E52")
        ax.set_xlabel("帧序")
        ax.set_ylabel("D (mm)")
        ax.set_title(f"逐帧 D（中位 {med:.2f} mm，虚线）；"
                     f"σ=帧内统计，散布大于 σ 即系统漂移")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        lay.addWidget(FigureCanvasQTAgg(fig))
        dlg.exec_()
