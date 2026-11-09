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

import numpy as np
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

    def __init__(self, files, px_um, lam, want_sigma=True, sens=0,
                 want_joint=False, want_mean=False, phase_hint=None,
                 spline_hint=None):
        super().__init__()
        self.files = files
        self.px_um = px_um
        self.lam = lam
        self.want_sigma = want_sigma
        self.sens = sens
        self.want_joint = want_joint
        self.want_mean = want_mean        # B 路线:弱衍射平均帧种子
        self.phase_hint = phase_hint      # mean 模式必选相
        self.spline_hint = spline_hint    # 可选探测器畸变 spline
        self.joint = None          # B13 联合解(done_all 后可用)
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

    def _load(self, f):
        img = load_any(f)
        return np.where(np.isfinite(img), img, 0.0).astype(np.float64)

    def _sniff_spline(self):
        """mean 模式:数据目录自动探测 distorsion*.spline。"""
        if self.spline_hint is not None:
            return self.spline_hint
        try:
            d = os.path.dirname(self.files[0]) if self.files else ""
            for x in os.listdir(d):
                if "distorsion" in x.lower() and x.lower().endswith(
                        ".spline"):
                    return os.path.join(d, x)
        except Exception:
            pass
        return None

    def run(self):
        n = len(self.files)
        if self.want_mean and self.lam and self.phase_hint and n >= 3:
            if self._run_mean_mode():
                self.done_all.emit()
                return
            # 种子失败 → 回退逐帧盲解(状态列会提示)
        cpts_pool = []                       # B13:(idx, ph, px_um, lam, c)
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
                    if self.want_joint:
                        try:
                            import solver_core as _sc
                            from torchfai.phases import phase_qs as _pqs
                            c = _sc.cpts_of_frame(
                                img, res["cx"], res["cy"], res["D_px"],
                                px_um * 1e-6, px_um * 1e-6, res["lam"],
                                _pqs(res["ph"]))
                            if c is not None:
                                cpts_pool.append((i, res["ph"], px_um,
                                                  res["lam"], c))
                        except Exception:
                            pass
            except Exception as e:
                d["status"] = f"异常: {e}"
            self.row.emit(i, d)
            self.prog.emit(i + 1, n)
        if self.want_joint and len(cpts_pool) >= 3 and not self.stop_flag:
            try:
                self.joint = self._run_joint(cpts_pool)
            except Exception:
                self.joint = None
        self.done_all.emit()

    def _run_mean_mode(self):
        """B 路线弱衍射模式(2026-11-08):全帧 mean → 一次种子 →
        每帧只跑 fit_tilt(免逐帧盲解,~10× 快)。种子失败返回 False
        (回退逐帧)。判决依据 _t_joint52:52 帧 mean+spline 种子
        Δ−0.13;8 帧 mean 信噪不足会崩 → 本模式只对 ≥3 帧开,种子
        质量由后续帧 MAD 审计兜底。"""
        import solver_core as _sc
        from torchfai.phases import phase_qs as _pqs
        from poni_out import make_pyfai_fwd as _mkfwd
        try:
            self.prog.emit(0, len(self.files) + 1)
            px_um = self.px_um
            if self._auto_px:
                m = sniff_meta(self.files[0])
                if "px" in m:
                    px_um = float(m["px"])
            px_m = px_um * 1e-6
            seed = _sc.b_route_mean_seed(
                (self._load(f) for f in self.files), px_m, px_m,
                self.lam, self.phase_hint, spline=self._sniff_spline())
            if seed is None:
                return False
            cx_n = seed["poni2"] / px_m - 0.5
            cy_n = seed["poni1"] / px_m - 0.5
            D_n = seed["dist"] / px_m
            # 种子层用 spline(画布拟合实证 Δ−0.13);控制点/联合层用
            # 自研 fwd——数据在线已校正,GT poni 的 spline 只是元数据,
            # 标定层再用 spline 会 rot/D 简并错位(_t_fwdspl2 铁证)。
            fwd = None
            qs = _pqs(self.phase_hint)
            self.prog.emit(1, len(self.files) + 1)
            cpts_pool = []
            for i, f in enumerate(self.files):
                if self.stop_flag:
                    return False
                d = {"file": f, "ph": self.phase_hint, "D_mm": "",
                     "m": "", "rms": "", "ps": "", "sD": "", "status": "",
                     "px": px_um}
                try:
                    img = self._load(f)
                    c = _sc.cpts_of_frame(img, cx_n, cy_n, D_n, px_m,
                                          px_m, self.lam, qs, fwd=fwd)
                    if c is None:
                        d["status"] = "⚠ 控制点不足(mean 种子)"
                    else:
                        d["D_mm"] = round(c["D"], 3)
                        d["rms"] = round((c["rms5"] or 0) * 1e3, 2)
                        d["status"] = "✓(mean 种子+fit_tilt)"
                        cpts_pool.append((i, self.phase_hint, px_um,
                                          self.lam, c))
                except Exception as e:
                    d["status"] = f"异常: {e}"
                self.row.emit(i, d)
                self.prog.emit(i + 2, len(self.files) + 1)
            if len(cpts_pool) >= 3 and not self.stop_flag:
                try:
                    self.joint = self._run_joint(cpts_pool)
                    if self.joint is not None and spl:
                        self.joint["spline"] = spl
                except Exception:
                    self.joint = None
            return True
        except Exception:
            return False

    def _run_joint(self, pool):
        """B13 联合标定:按 (相, px) 分组取最大组(静态装置假设),
        组内全局 (D,rot) + 每帧中心联合 LSQ + bootstrap σ。"""
        import solver_core as _sc
        from collections import Counter
        grp = Counter((ph, round(px, 3)) for _, ph, px, _, _ in pool)
        (ph, px3), _ = grp.most_common(1)[0]
        sub = [(i, lam, c) for i, ph_, px, lam, c in pool
               if ph_ == ph and round(px, 3) == px3]
        if len(sub) < 3:
            return None
        px_m = px3 * 1e-3
        res = _sc.joint_solve_frames([c for _, _, c in sub], px_m, px_m,
                                     boot=6)
        if res is None:
            return None
        Df = np.array([c["D"] for _, _, c in sub if c.get("D")])
        if len(Df) >= 3:
            res["D_mad"] = float(np.median(
                np.abs(Df - np.median(Df))))
            res["D_spread"] = float(Df.max() - Df.min())
        else:
            res["D_mad"] = None
        res["ph"] = ph
        res["px_um"] = px3
        res["lam"] = sub[0][1]
        res["files"] = [self.files[i] for i, _, _ in sub]
        res["idx"] = [i for i, _, _ in sub]
        return res


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
        par.addWidget(QLabel("λ/能量:"))
        self.e_lam = QLineEdit(lam or ""); self.e_lam.setMaximumWidth(90)
        self.e_lam.setPlaceholderText("空=联立解")
        self.e_lam.setToolTip("按右侧单位填波长(Å)或光子能量(keV)")
        par.addWidget(self.e_lam)
        # 单位切换(2026-11-07):与主窗口同款,keV↔Å 换算 12.39842
        self.cb_lamu = QComboBox(); self.cb_lamu.addItems(["Å", "keV"])
        self.cb_lamu.currentIndexChanged.connect(self._on_lamu)
        par.addWidget(self.cb_lamu)
        par.addWidget(QLabel("灵敏度:"))
        self.cb_sens = QComboBox(); self.cb_sens.addItems(["标准", "高", "极高"])
        par.addWidget(self.cb_sens)
        self.cb_sigma = QCheckBox("σ(D)"); self.cb_sigma.setChecked(True)
        self.cb_sigma.setToolTip("每帧环级 bootstrap(约 +1~3s/帧)")
        par.addWidget(self.cb_sigma)
        self.cb_joint = QCheckBox("🔗 联合标定")
        self.cb_joint.setChecked(True)
        self.cb_joint.setToolTip(
            "B13 多帧联合(≥3 帧同相同 px):全局 D/rot + 每帧中心,\n"
            "σ(D)=√N 收敛(51 帧实证 0.010mm);解完表格底行显示 JOINT")
        par.addWidget(self.cb_joint)
        self.cb_mean = QCheckBox("🏠 弱衍射·平均帧种子")
        self.cb_mean.setToolTip(
            "B 路线(静态装置弱衍射,如 alumina):全帧平均 → 一次圆解\n"
            "+spline fit_tilt 做种子 → 每帧只跑倾角拟合(约 10× 快)。\n"
            "需 λ+标样相;目录内 distorsion*.spline 自动探测。\n"
            "逐帧盲解次优盆丛生的数据集用此模式(alumina 5/8 帧盆)")
        self.cb_mean.stateChanged.connect(self._on_mean_toggle)
        par.addWidget(self.cb_mean)
        from torchfai.phases import PHASE_NAMES as _PN
        self.cb_phase = QComboBox()
        self.cb_phase.addItems(list(_PN))
        self.cb_phase.setEnabled(False)
        self.cb_phase.setToolTip("平均帧种子模式必选标样相")
        par.addWidget(self.cb_phase)
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
        self.b_jponi = QPushButton("🔗 联合 .poni"); self.b_jponi.setEnabled(False)
        self.b_jponi.setToolTip("联合解(全局 D/rot+中位中心)导出 joint.poni")
        self.b_jponi.clicked.connect(self.export_joint_poni)
        self.b_drift = QPushButton("📈 D 漂移曲线"); self.b_drift.setEnabled(False)
        self.b_drift.clicked.connect(self.show_drift)
        self.b_close = QPushButton("关闭"); self.b_close.clicked.connect(self.accept)
        for b in (self.b_csv, self.b_poni, self.b_jponi, self.b_drift,
                  self.b_close):
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
    HC_KEV_A = 12.39842

    def _on_lamu(self, i):
        """单位切换:Å↔keV 换算框内数值(物理量保持;两方向同式)。"""
        t = self.e_lam.text().strip()
        if not t:
            return
        try:
            v = float(t)
        except ValueError:
            return
        self.e_lam.setText(f"{self.HC_KEV_A / v:.10g}")

    def _lam_A(self):
        """→ 波长 Å(float)或 None。keV 模式自动换算。"""
        t = self.e_lam.text().strip()
        if not t:
            return None
        try:
            v = float(t)
        except ValueError:
            return None
        return self.HC_KEV_A / v if (v > 0 and
                self.cb_lamu.currentIndex() == 1) else (v if v > 0 else None)

    def run_batch(self):
        if not self.files:
            QMessageBox.information(self, "批量", "先选文件夹或文件。")
            return
        use_mean = self.cb_mean.isChecked()
        if use_mean:
            if not self._lam_A():
                QMessageBox.warning(
                    self, "平均帧种子", "该模式需要 λ(波长/能量)。")
                return
            if len(self.files) < 3:
                QMessageBox.warning(
                    self, "平均帧种子", "该模式需要 ≥3 帧(静态装置)。")
                return
        px = float(self.e_px.text() or 172.0)
        import solver_core as _sc
        _sc.set_ring_sensitivity(self.cb_sens.currentIndex())
        self.worker = BatchWorker(self.files, px, self._lam_A(),
                                  want_sigma=self.cb_sigma.isChecked(),
                                  sens=self.cb_sens.currentIndex(),
                                  want_joint=self.cb_joint.isChecked(),
                                  want_mean=use_mean,
                                  phase_hint=(self.cb_phase.currentText()
                                              if use_mean else None))
        self.worker._auto_px = self.cb_pxauto.isChecked()
        self.table.setRowCount(len(self.files))
        self.rows = [None] * len(self.files)
        self.worker.row.connect(self.on_row)
        self.worker.prog.connect(self.on_prog)
        self.worker.done_all.connect(self.on_done_all)
        self.b_run.setEnabled(False); self.b_stop.setEnabled(True)
        rng = len(self.files) + 1 if use_mean else len(self.files)
        self.pb.setRange(0, rng); self.pb.setValue(0)
        self.pb.setVisible(True)
        self.worker.start()

    def _on_mean_toggle(self, state):
        """平均帧种子模式联动:相下拉启用/禁用。"""
        on = state == 2          # Qt.Checked
        self.cb_phase.setEnabled(on)

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
        j = self.worker.joint if self.worker else None
        self.b_jponi.setEnabled(j is not None)
        msg = (f"{len(ok)}/{len(self.rows)} 帧解出。"
               + ("漂移曲线可用（≥5 帧）。" if len(ok) >= 5 else ""))
        if j is not None:
            Dm = j["dist"] * 1e3
            sg = j.get("sigma_boot")
            mad = j.get("D_mad")
            self.table.setRowCount(len(self.files) + 1)
            warn = (";⚠帧D散布大(MAD %.2fmm,种子盆地风险)" % mad
                    if mad and mad > 1.0 else "")
            vals = ["🔗 JOINT", j["ph"], round(Dm, 3), f"{j['nf']}帧",
                    round(j["rms"] * 1e3, 2), "",
                    (round(sg, 3) if sg else ""),
                    (f"σ={'±%.3f' % sg}mm" if sg else "")
                    + (f";帧D MAD={mad:.2f}mm" if mad is not None
                       else "") + warn]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setForeground(Qt.darkRed if warn else Qt.darkBlue)
                self.table.setItem(len(self.files), c, it)
            msg += (f"\n联合标定:D={Dm:.3f}mm"
                    + (f" ±{sg:.3f}mm" if sg else "")
                    + f"(rot1={np.degrees(j['rot1']):+.2f}°,"
                      f"{j['nf']} 帧 {j['ph']})"
                    + (f"\n审计:帧独立 D 的 MAD={mad:.2f}mm"
                       + (" ⚠ 散布大→部分帧种子落次优盆,建议核对"
                          " 漂移曲线剔除离群帧后重解" if warn else "")
                       if mad is not None else ""))
        else:
            self.table.setRowCount(len(self.files))
            if self.cb_joint.isChecked():
                msg += "\n联合标定:有效帧 <3(或控制点不足),未成解"
        QMessageBox.information(self, "批量完成", msg)

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
            j = self.worker.joint if self.worker else None
            if j is not None:
                wr.writerow(["JOINT", j["ph"], round(j["dist"] * 1e3, 4),
                             "", round(j["rms"] * 1e3, 3), "",
                             (round(j["sigma_boot"], 4)
                              if j.get("sigma_boot") else ""),
                             f"nf={j['nf']} rot1="
                             f"{np.degrees(j['rot1']):+.4f}",
                             j["px_um"]])
        QMessageBox.information(self, "CSV", f"已导出 {p}")

    def export_joint_poni(self):
        """B13 联合解导出 poni(全局 D/rot + 帧间中位中心)。"""
        j = self.worker.joint if self.worker else None
        if j is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "导出联合 .poni",
                                           "joint.poni", "PONI (*.poni)")
        if not p:
            return
        import numpy as _np
        p1 = float(_np.median(j["p1s"]))
        p2 = float(_np.median(j["p2s"]))
        px = j["px_um"] * 1e-6
        lam = j["lam"] * 1e-10
        txt = (f"# Joint calibration ({j['nf']} frames, {j['ph']})\n"
               f"# sigma_D = "
               f"{('%.4f' % j['sigma_boot']) if j.get('sigma_boot') else 'n/a'}"
               f" mm (bootstrap)\n"
               f"PixelSize1: {px}\nPixelSize2: {px}\n"
               f"Distance: {j['dist']}\n"
               f"Poni1: {p1}\nPoni2: {p2}\n"
               f"Rot1: {j['rot1']}\nRot2: {j['rot2']}\nRot3: 0.0\n"
               f"Wavelength: {lam}\n")
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
        QMessageBox.information(
            self, "联合 poni",
            f"已导出 {p}\nD={j['dist']*1e3:.3f}mm "
            f"中心=({p2/px-0.5:.0f},{p1/px-0.5:.0f})px "
            f"rot=({np.degrees(j['rot1']):+.3f},"
            f"{np.degrees(j['rot2']):+.3f})°")

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
        j = self.worker.joint if self.worker else None
        ttl = (f"逐帧 D（中位 {med:.2f} mm，虚线）；"
               f"σ=帧内统计，散布大于 σ 即系统漂移")
        if j is not None:
            Dj = j["dist"] * 1e3
            sj = j.get("sigma_boot")
            ax.axhspan(Dj - (sj or 0), Dj + (sj or 0),
                       color="#4C72B0", alpha=0.18)
            ax.axhline(Dj, lw=1.4, color="#4C72B0")
            ttl += (f"；🔗 联合 D={Dj:.3f}"
                    + (f"±{sj:.3f}" if sj else "") + " mm（蓝带）")
        ax.set_xlabel("帧序")
        ax.set_ylabel("D (mm)")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        lay.addWidget(FigureCanvasQTAgg(fig))
        dlg.exec_()
