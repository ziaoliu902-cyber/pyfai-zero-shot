# -*- coding: utf-8 -*-
"""gui_wizard.py：一站流傻瓜向导（2026-11-02，T-GUI-WIZ；i18n 2026-11-03）。

 五步 QWizard：①拖图/选图 → ②参数确认（自动嗅探已填好，只按下一步）
 → ③自动求解（后台线程）→ ④自动标样自检（判语大灯）→ ⑤导出 poni/
 复制结果/再来一张。复用 gui_solver 的全部零件（load_any/
 sniff_meta/SolveWorker/CheckWorker/check_std/hybrid_ai），零口径分叉。
 多语言：全部文案走 i18n.tr；🌐 键切换 zh/en，切换=换语言+重建窗口
 （ui 无状态化；求解中切换=新会话，后台线程对已删页面自动断连）。
 启动：python gui_wizard.py
"""
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QDoubleValidator, QFont
from PyQt5.QtWidgets import (
    QApplication, QWizard, QWizardPage, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QComboBox)

from gui_solver import (load_any, sniff_meta, SolveWorker, CheckWorker,
                        check_std, hybrid_ai, np, apply_spline,
                        set_splinefile, EscalationFlow)
import theme
from i18n import tr, get_lang, set_lang, LANG_NAMES

IMG_FILTER = ("衍射图 (*.edf *.cbf *.h5 *.hdf5 *.tif *.tiff *.mar* *.png *.jpg)"
              ";;所有 (*.*)")

_CUR = None   # 当前窗口的保活引用：无父 QWidget 必须有 Python 引用，
              # 否则被 GC → 窗口销毁 → 程序退出（pythonw 下表现为闪退）


class PageImg(QWizardPage):
    """①选择图像：拖进窗口或点按钮；可选 .spline 畸变校正（可撤销）。"""

    def __init__(self):
        super().__init__()
        self.setTitle(tr("p1_title"))
        self.setSubTitle(tr("p1_sub"))
        self.path = None
        self.paths = None       # 多帧模式(≥2 张)文件列表,2026-11-08
        self.img = None
        self.img_raw = None    # 校正前的原图（撤销用）
        self.spline = None     # 已应用的 .spline 路径（None=未校正）
        lay = QVBoxLayout()
        self.drop = QLabel(tr("p1_drop"))
        self.drop.setAlignment(Qt.AlignCenter)
        self.drop.setMinimumHeight(160)
        self.drop.setAcceptDrops(True)
        self.drop.dragEnterEvent = self._enter
        self.drop.dropEvent = self._drop
        row = QHBoxLayout()
        self.b = QPushButton(tr("p1_browse"))
        self.b.setProperty("primary", True)
        self.b.clicked.connect(self._pick)
        self.b_sp = QPushButton(tr("sp_btn"))
        self.b_sp.clicked.connect(self._pick_sp)
        self.b_sp_off = QPushButton(tr("sp_off_btn"))
        self.b_sp_off.clicked.connect(self._undo_sp)
        for b in (self.b, self.b_sp, self.b_sp_off):
            row.addWidget(b)
        self.info = QLabel("")
        self.info.setProperty("muted", True)
        self.info.setWordWrap(True)
        self.sp_lab = QLabel("")
        self.sp_lab.setProperty("muted", True)
        self.sp_lab.setWordWrap(True)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        lay.addWidget(self.drop)
        lay.addLayout(row)
        lay.addWidget(self.info)
        lay.addWidget(self.sp_lab)
        self.setLayout(lay)
        self._sp_refresh()

    def initializePage(self):
        # 拖放框配色走主题令牌（spec §2：颜色不硬编码）
        t = theme.TOKENS[self.wizard().theme_name]
        self.drop.setStyleSheet(
            "border: 2px dashed %s; border-radius: 6px;"
            "background: %s; color: %s; font-size: 15px;"
            % (t["border"], t["surface"], t["muted"]))

    def _enter(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def _drop(self, e):
        urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self._load_many([u.toLocalFile() for u in urls])

    def _pick(self):
        ps, _ = QFileDialog.getOpenFileNames(self, tr("p1_pick"), "",
                                             IMG_FILTER)
        if ps:
            self._load_many(ps)

    def _load_many(self, ps):
        """单张=经典盲解;≥2 张=🏠多帧平均模式(弱衍射专用,2026-11-08)。

        预览/嗅探取第 1 张;worker 层流式 mean+逐帧控制点+联合 LSQ。
        """
        try:
            img0 = load_any(ps[0])
        except Exception as e:
            QMessageBox.warning(self, tr("p1_openfail"), str(e))
            return
        self.path = ps[0]
        self.paths = list(ps)
        self.img = img0
        self.img_raw = None          # 新图=全新会话，样条状态清零（F6）
        self.spline = None
        if len(ps) >= 2:
            self.info.setText(tr("p1_multi", n=len(ps),
                                 name=os.path.basename(ps[0]),
                                 w=img0.shape[1], h=img0.shape[0]))
        else:
            self.info.setText(tr("p1_loaded", name=os.path.basename(ps[0]),
                                 w=img0.shape[1], h=img0.shape[0]))
        self._sp_refresh()
        self.completeChanged.emit()

    def _load(self, p):
        self._load_many([p])

    # ── 畸变样条（T-SPL）：F1 入口 / F2 可见可撤销 / F5 边带置零在引擎侧 ──
    def _pick_sp(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("sp_pick"), "",
                                           tr("sp_filter"))
        if p:
            self._do_sp(p)

    def _do_sp(self, p):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            base = self.img_raw if self.img_raw is not None else self.img
            corrected = apply_spline(base, p)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, tr("sp_fail"), str(e))
            return
        QApplication.restoreOverrideCursor()
        if self.img_raw is None:
            self.img_raw = self.img
        self.img = corrected
        self.spline = p
        self._sp_refresh()
        self.completeChanged.emit()

    def _undo_sp(self):
        if self.img_raw is not None:
            self.img = self.img_raw
            self.img_raw = None
        self.spline = None
        self._sp_refresh()
        self.completeChanged.emit()

    def _sp_refresh(self):
        on = self.img is not None
        self.b_sp.setVisible(on)
        self.b_sp_off.setVisible(on and self.spline is not None)
        self.sp_lab.setText(
            tr("sp_on", name=os.path.basename(self.spline))
            if self.spline else "")

    def isComplete(self):
        return self.img is not None


class PageParam(QWizardPage):
    """②参数确认：像素/波长自动嗅探填好；用户只需按下一步。"""

    def __init__(self):
        super().__init__()
        self.setTitle(tr("p2_title"))
        self.setSubTitle(tr("p2_sub"))
        g = QGridLayout()
        self.e_px = QLineEdit("172")
        self.e_px.setValidator(QDoubleValidator())
        self.e_lam = QLineEdit("")
        self.e_lam.setValidator(QDoubleValidator())
        g.addWidget(QLabel(tr("p2_px")), 0, 0)
        g.addWidget(self.e_px, 0, 1)
        g.addWidget(QLabel(tr("p2_px_hint")), 0, 2)
        g.addWidget(QLabel(tr("p2_lam")), 1, 0)
        g.addWidget(self.e_lam, 1, 1)
        # 单位切换(2026-11-07):与主窗口同款,Å↔keV(λ=12.39842/E)
        self.cb_lamu = QComboBox(); self.cb_lamu.addItems(["Å", "keV"])
        self.cb_lamu.currentIndexChanged.connect(self._on_lamu)
        g.addWidget(self.cb_lamu, 1, 2)
        g.addWidget(QLabel(tr("p2_lam_hint")), 1, 3)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        g.addWidget(self.note, 2, 0, 3, 1)
        # 多帧模式的标样相下拉(2026-11-08):mean 种子锁相必选
        from torchfai.phases import PHASE_NAMES as _PN
        self.cb_phase = QComboBox()
        self.cb_phase.addItems(list(_PN))
        self.cb_phase.setVisible(False)
        self.ph_lab = QLabel(tr("p2_phase"))
        self.ph_lab.setVisible(False)
        g.addWidget(self.ph_lab, 3, 0)
        g.addWidget(self.cb_phase, 3, 1)
        self.setLayout(g)

    def initializePage(self):
        wiz = self.wizard()
        multi = wiz.p1.paths is not None and len(wiz.p1.paths) >= 2
        self.cb_phase.setVisible(multi)
        self.ph_lab.setVisible(multi)
        meta = sniff_meta(wiz.p1.path)
        if "px" in meta:
            self.e_px.setText(f"{meta['px']:g}")
        else:
            self.e_px.setText("172")
        if "lam" in meta:
            # 文件头波长是 Å;keV 模式换算成能量填入(物理量一致)
            v = meta["lam"]
            if self.cb_lamu.currentIndex() == 1:
                self.e_lam.setText(f"{self.HC_KEV_A / v:.10g}")
            else:
                self.e_lam.setText(f"{v:g}")
        else:
            self.e_lam.setText("")
        msg = []
        if meta["src"]:
            lam = tr("p2_auto_lam", lamv=f"{meta['lam']:g}") if "lam" in meta else ""
            msg.append(tr("p2_auto", src=meta["src"],
                          px=meta.get("px", "—"), lam=lam))
            if "D_mm" in meta:
                msg.append(tr("p2_hdr_d", d=f"{meta['D_mm']:.1f}"))
        else:
            msg.append(tr("p2_nopx"))
        self.note.setText("\n".join(msg))

    # ---- λ/能量统一口径(2026-11-07,与主窗口同款) ----
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
        """→ 波长 Å(float)或 None(空=联立解)。keV 模式自动换算。"""
        t = self.e_lam.text().strip()
        if not t:
            return None
        try:
            v = float(t)
        except ValueError:
            return None
        if v <= 0:
            return None
        return self.HC_KEV_A / v if self.cb_lamu.currentIndex() == 1 else v

    def validatePage(self):
        wiz = self.wizard()
        multi = wiz.p1.paths is not None and len(wiz.p1.paths) >= 2
        try:
            v = float(self.e_px.text())
            assert v > 1
        except Exception:
            QMessageBox.warning(self, tr("p2_badpx_title"), tr("p2_badpx"))
            return False
        if multi and self._lam_A() is None:
            # 多帧=窄角域弱衍射专用,λ–D 简并破不了(σ39% 实证),必填
            QMessageBox.warning(self, tr("p2_title"),
                                tr("p2_needlam_multi"))
            return False
        return True


class JointWorker(QThread):
    """多帧弱衍射求解(2026-11-08):复用批量模式同款零分叉口径。

    链:b_route_mean_seed(流式平均+锁相圆解+fit_tilt 种子) → 逐帧
    cpts_of_frame(fwd=None,数据在线已校正) → joint_solve_frames
    (全局 D/rot+每帧中心,bootstrap σ)。与 batch_gui._run_joint 同源,
    种子层 spline=None(向导数据已被 apply_spline 校正)。
    """
    done = pyqtSignal(object)
    prog = pyqtSignal(str)

    def __init__(self, paths, lam_A, phase, px_um, spline=None):
        super().__init__()
        self.paths = list(paths)
        self.lam = lam_A
        self.phase = phase
        self.px_um = px_um
        self.spline = spline      # 仅元数据(导出 poni 带 SplineFile 用)

    def run(self):
        import numpy as np
        import solver_core as _sc
        from torchfai.phases import phase_qs as _pqs
        try:
            px_m = self.px_um * 1e-6
            self.prog.emit(tr("jw_seed", n=len(self.paths)))
            seed = _sc.b_route_mean_seed(
                (load_any(f) for f in self.paths), px_m, px_m,
                self.lam, self.phase)
            if seed is None:
                self.done.emit(None)
                return
            cx_n = seed["poni2"] / px_m - 0.5
            cy_n = seed["poni1"] / px_m - 0.5
            D_n = seed["dist"] / px_m
            qs = _pqs(self.phase)
            cpts = []
            Ds = []
            for i, f in enumerate(self.paths):
                self.prog.emit(tr("jw_frame", i=i + 1,
                                  n=len(self.paths)))
                try:
                    img = load_any(f)
                    c = _sc.cpts_of_frame(img, cx_n, cy_n, D_n, px_m,
                                          px_m, self.lam, qs)
                    if c is not None:
                        cpts.append(c)
                        if c.get("D"):
                            Ds.append(c["D"])
                except Exception:
                    continue
            if len(cpts) < 3:
                self.done.emit(None)
                return
            self.prog.emit(tr("jw_joint"))
            j = _sc.joint_solve_frames(cpts, px_m, px_m, boot=6)
            if j is None:
                self.done.emit(None)
                return
            j["ph"] = self.phase
            j["lam"] = self.lam
            j["px_um"] = self.px_um
            j["nf_in"] = len(self.paths)
            j["spline"] = self.spline
            if len(Ds) >= 3:
                j["D_mad"] = float(np.median(
                    np.abs(np.array(Ds) - np.median(Ds))))
            self.done.emit(j)
        except Exception:
            import traceback
            traceback.print_exc()
            self.done.emit(None)


class PageSolve(QWizardPage):
    """③自动求解：进入页面即开跑，完成才能下一步。"""

    def __init__(self):
        super().__init__()
        self.setTitle(tr("p3_title"))
        self.setSubTitle(tr("p3_sub"))
        self._ok = False
        self.worker = None
        self.flow = None          # 环不足补救(2026-11-07 与主窗口共享状态机)
        lay = QVBoxLayout()
        self.pb = QProgressBar(); self.pb.setRange(0, 0)
        self.lab = QLabel(tr("p3_ready"))
        self.lab.setWordWrap(True)
        self.out = QTextEdit(); self.out.setReadOnly(True)
        lay.addWidget(self.pb)
        lay.addWidget(self.lab)
        lay.addWidget(self.out, 1)
        self.setLayout(lay)

    def _ensure_flow(self):
        """补救控制器晚建:要引用向导兄弟页的控件。"""
        if self.flow is not None:
            return
        wiz = self.wizard()

        def retry():
            QTimer.singleShot(60, self._restart)

        self.flow = EscalationFlow(
            self,
            get_lam=lambda: wiz.p2.e_lam.text().strip(),
            set_lam=lambda v: wiz.p2.e_lam.setText(v),
            retry=retry,
            log=lambda m: self.out.append(m),
            status=lambda m: self.lab.setText(m))

    def _restart(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self._start()

    def initializePage(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self._start()

    def _start(self):
        wiz = self.wizard()
        self._ensure_flow()
        self._ok = False
        self.pb.setVisible(True)
        lam_t = wiz.p2._lam_A()
        multi = wiz.p1.paths is not None and len(wiz.p1.paths) >= 2
        if multi:
            # 多帧弱衍射:mean 种子+联合(与批量模式同口径,2026-11-08)
            self.worker = JointWorker(wiz.p1.paths, lam_t,
                                      wiz.p2.cb_phase.currentText(),
                                      float(wiz.p2.e_px.text()),
                                      spline=wiz.p1.spline)
            self.worker.done.connect(self._done_joint)
            self.worker.prog.connect(self.lab.setText)
            self.lab.setText(tr("p3_running"))
            self.worker.start()
            return
        self.worker = SolveWorker(wiz.p1.img,
                                  lam_t,
                                  float(wiz.p2.e_px.text()))
        self.worker.done.connect(self._done)
        self.worker.fail.connect(self._fail)
        self.lab.setText(tr("p3_running"))
        self.worker.start()

    def _done_joint(self, j):
        """多帧联合结果 → 向导 res_j;p4 渲染联合一致性大灯。"""
        self.pb.setVisible(False)
        wiz = self.wizard()
        wiz.res = None
        wiz.res_j = j
        if j is None:
            self.lab.setText(tr("p3_none"))
            self.out.setPlainText(tr("jw_fail"))
            self._ok = False
            self.completeChanged.emit()
            return
        sig = j.get("sigma_boot")
        sig_s = f"{sig:.3f}" if sig is not None else "n/a"
        mad = j.get("D_mad")
        mad_s = f"{mad:.3f}" if mad is not None else "n/a"
        self.out.setPlainText("\n".join([
            tr("jw_phase", ph=j["ph"]),
            tr("jw_frames", nf=j["nf"], n_in=j.get("nf_in", j["nf"])),
            tr("s_lam", v=f"{j['lam']:.4f}"),
            tr("jw_dist", d=f"{j['dist']*1e3:.3f}", sig=sig_s),
            tr("jw_mad", mad=mad_s),
            tr("jw_npts", n=j["npts"], rms=f"{j['rms']*1e3:.1f}")]))
        self.lab.setText(tr("jw_done"))
        self._ok = True
        self.completeChanged.emit()

    def _fail(self, tb):
        self.pb.setVisible(False)
        if self.flow is not None:
            self.flow.on_fail()             # 爬升中异常 → 恢复档位
        self.lab.setText(tr("p3_fail"))
        self.out.setPlainText(tb)

    def _done(self, res):
        self.pb.setVisible(False)
        wiz = self.wizard()
        wiz.res = res
        if res is None:
            climbing = self.flow._esc_next is not None
            self.flow.on_none()             # 爬升中→下一档;首次→问询
            if self.flow._esc_next is None and not climbing:
                self.lab.setText(tr("p3_none"))
                self.out.setPlainText(tr("p3_honest"))
                self.completeChanged.emit()
            return
        self.flow.on_success()
        px_um = float(wiz.p2.e_px.text())
        D_mm = res["D_px"] * px_um * 1e-3
        tf = res.get("tf")
        tilt = (tr("s_tilt_v", a=tf["rot1"] * 1e3, b=tf["rot2"] * 1e3)
                if tf else tr("s_tilt_none"))
        from torchfai.phases import HARMONIC_PHASES
        _m = res["sol"][3]
        _harm = (res["ph"] in HARMONIC_PHASES and _m <= 4)
        self.out.setPlainText(
            "\n".join([
                tr("s_phase", ph=res["ph"]),
                tr("s_lam", v=f"{res['lam']:.4f}"),
                tr("s_dist", v=f"{D_mm:.2f}"),
                tr("s_center", x=f"{res['cx']:.1f}", y=f"{res['cy']:.1f}"),
                tr("s_match", m=_m, rms=f"{res['sol'][4]:.2f}",
                   ps=f"{res['sol'][6]:.3f}"),
                tr("s_tilt", v=tilt)]
                + ([tr("p3_harmwarn", ph=res["ph"], a=int(_m), d1=f"{D_mm:.0f}",
                       d2=f"{D_mm*2:.0f}", d3=f"{D_mm*3:.0f}")]
                   if _harm else [])))
        self.flow.on_success()      # 爬档解出注记(在正式文本后追加,不覆盖)
        # 自动 σ(D)(2026-11-07 A4):后台快速档(环级,不含倾角),完成后追加
        try:
            self.uncw = UncWorker(wiz.p1.img, res,
                                  float(wiz.p2.e_px.text() or 172.0),
                                  b_tilt=0)
            self.uncw.done.connect(self._unc_done)
            self.uncw.fail.connect(lambda tb: None)
            self.uncw.start()
        except Exception:
            self.uncw = None
        warn = "" if wiz.p2._lam_A() is not None else tr("p3_lamwarn")
        self.lab.setText(tr("p3_done") + warn)
        self._ok = res is not None
        self.completeChanged.emit()

    def _unc_done(self, u):
        """σ(D) 追加到 p3 输出(页面可能已被重建,全 try 兜底)。"""
        try:
            bs = u["bs"]
            if bs is None or bs[0] == "few":
                return
            s_, lo_, hi_, n_ = bs
            self.out.append(tr("s_unc", s=f"{s_:.2f}", lo=f"{lo_:.1f}",
                               hi=f"{hi_:.1f}", n=n_))
        except Exception:
            pass

    def isComplete(self):
        return self._ok


class PageCheck(QWizardPage):
    """④自动标样自检：进入即跑，判语大灯。"""

    def __init__(self):
        super().__init__()
        self.setTitle(tr("p4_title"))
        self.setSubTitle(tr("p4_sub"))
        self._ok = False
        self.worker = None
        lay = QVBoxLayout()
        self.big = QLabel("…")
        self.big.setAlignment(Qt.AlignCenter)
        self.big.setWordWrap(True)
        f = QFont(); f.setPointSize(14); f.setBold(True)
        self.big.setFont(f)
        self.out = QTextEdit(); self.out.setReadOnly(True)
        lay.addWidget(self.big)
        lay.addWidget(self.out, 1)
        self.setLayout(lay)

    def initializePage(self):
        if self.worker is not None and self.worker.isRunning():
            return
        wiz = self.wizard()
        if getattr(wiz, "res_j", None) is not None:
            self._render_joint(wiz.res_j)      # 多帧:联合一致性大灯
            return
        self.worker = CheckWorker(wiz.p1.img, wiz.res,
                                  float(wiz.p2.e_px.text()))
        self.worker.done.connect(self._done)
        self.worker.fail.connect(self._fail)
        self.big.setText(tr("p4_running"))
        self.worker.start()

    def _fail(self, tb):
        self.big.setText(tr("p4_fail"))
        self.out.setPlainText(tb)
        self._ok = True
        self.completeChanged.emit()

    def _render_joint(self, j):
        """多帧联合结果④:一致性大灯(2026-11-08)。

        大灯:帧间 D MAD <1mm 绿(静态装置一致性);否则黄(帧解散布大
        =样品动过/混入异构数据)。置信度:多帧联合本身高可靠(互为
        校验);单张弱衍射才需要"建议复核"提示(_done 分支)。
        """
        t = theme.TOKENS[self.wizard().theme_name]
        mad = j.get("D_mad")
        good = mad is None or mad < 1.0
        sig = j.get("sigma_boot")
        sig_s = f"{sig:.3f}" if sig is not None else "n/a"
        lines = [tr("p4j_title", nf=j["nf"]),
                 tr("jw_dist", d=f"{j['dist']*1e3:.3f}", sig=sig_s),
                 tr("jw_mad", mad=(f"{mad:.3f}" if mad is not None
                                   else "n/a")),
                 tr("p4j_note")]
        if not good:
            lines.append(tr("p4j_madbad"))
        self.out.setPlainText("\n".join(lines))
        self.big.setText(tr("p4j_ok") if good else tr("p4j_warn"))
        self.big.setStyleSheet(
            "color: %s;" % (t["good"] if good else t["warn"]
                            if "warn" in t else t["danger"]))
        self._ok = True
        self.completeChanged.emit()

    def _done(self, c):
        self.out.setPlainText("\n".join(c["lines"]))
        # ②置信度分级(2026-11-08):弱衍射单张"解出但错"(次优盆)无自
        # 证——用 m/ps 的组合给分级提示,不承诺判对,只标"绿灯多绿"。
        wiz = self.wizard()
        res = wiz.res
        if res is not None:
            m_, ps_ = int(res["sol"][3]), float(res["sol"][6])
            lam_given = wiz.p2._lam_A() is not None
            if m_ <= 5 or ps_ < 0.35:
                tier, col = tr("c_tier_low"), "danger"
            elif m_ <= 8 or ps_ < 0.55 or not lam_given:
                tier, col = tr("c_tier_mid"), "warn"
            else:
                tier, col = tr("c_tier_high"), "good"
            t = theme.TOKENS[wiz.theme_name]
            self.out.append("")
            self.out.append(tr("c_tier_line", tier=tier))
            if col == "danger":
                self.out.append(tr("c_tier_multi_hint"))
        t = theme.TOKENS[self.wizard().theme_name]
        good = c["dev_max"] is not None and c["dev_max"] < 0.05
        self.big.setText(c["verdict"])
        self.big.setStyleSheet("color: %s;" % (t["good"] if good else t["danger"]))
        self._ok = True
        self.completeChanged.emit()

    def isComplete(self):
        return self._ok


class PageDone(QWizardPage):
    """⑤导出收尾：poni / 复制 / 再来一张。"""

    def __init__(self):
        super().__init__()
        self.setTitle(tr("p5_title"))
        self.setSubTitle(tr("p5_sub"))
        lay = QVBoxLayout()
        self.sum = QTextEdit(); self.sum.setReadOnly(True)
        row = QHBoxLayout()
        b1 = QPushButton(tr("b_poni"))
        b1.setProperty("primary", True)
        b1.clicked.connect(self._save)
        b2 = QPushButton(tr("b_copy"))
        b2.clicked.connect(self._copy)
        b3 = QPushButton(tr("b_again"))
        b3.clicked.connect(self._restart)
        for b in (b1, b2, b3):
            row.addWidget(b)
        lay.addWidget(self.sum, 1)
        lay.addLayout(row)
        self.setLayout(lay)

    def _restart(self):
        """回到第 1 步换下一张图：清全部状态，不退出程序。"""
        wiz = self.wizard()
        wiz.res = None
        wiz.res_j = None
        p1 = wiz.p1
        p1.path = None
        p1.paths = None
        p1.img = None
        p1.img_raw = None
        p1.spline = None
        p1._sp_refresh()
        p1.info.setText("")
        p1.completeChanged.emit()
        wiz.p3._ok = False
        wiz.p3.out.setPlainText("")
        wiz.p3.lab.setText(tr("p3_ready"))
        wiz.p4._ok = False
        wiz.p4.out.setPlainText("")
        wiz.p4.big.setText("…")
        self.sum.setPlainText("")
        wiz.restart()

    def initializePage(self):
        wiz = self.wizard()
        j = getattr(wiz, "res_j", None)
        if j is not None:
            sig = j.get("sigma_boot")
            self.sum.setPlainText("\n".join([
                tr("p5j_frames", n=j.get("nf_in", j["nf"])),
                tr("p5_image", name=os.path.basename(wiz.p1.path)),
                tr("jw_phase", ph=j["ph"]),
                tr("s_lam", v=f"{j['lam']:.4f}"),
                tr("jw_dist", d=f"{j['dist']*1e3:.3f}",
                   sig=(f"{sig:.3f}" if sig is not None else "n/a")),
                tr("jw_mad", mad=(f"{j['D_mad']:.3f}"
                                  if j.get("D_mad") is not None
                                  else "n/a")),
                tr("jw_npts", n=j["npts"], rms=f"{j['rms']*1e3:.1f}")]
               + ([tr("sp_line", name=os.path.basename(wiz.p1.spline))]
                  if wiz.p1.spline else [])
               + ["", tr("p5_outpath")]))
            return
        res = wiz.res
        px_um = float(wiz.p2.e_px.text())
        D_mm = res["D_px"] * px_um * 1e-3
        self.sum.setPlainText(
            "\n".join([
                tr("p5_image", name=os.path.basename(wiz.p1.path)),
                tr("p5_line2", ph=res["ph"], lam=f"{res['lam']:.4f}",
                   d=f"{D_mm:.2f}"),
                tr("p5_line3", x=f"{res['cx']:.1f}", y=f"{res['cy']:.1f}",
                   m=res["sol"][3], rms=f"{res['sol'][4]:.2f}",
                   ps=f"{res['sol'][6]:.3f}"),
                tr("p5_tilt", v=(tr("yes") if res.get("tf") else tr("no")))]
               + ([tr("sp_line", name=os.path.basename(wiz.p1.spline))]
                  if wiz.p1.spline else [])
               + ["", tr("p5_outpath")]))

    def _save(self):
        wiz = self.wizard()
        j = getattr(wiz, "res_j", None)
        default = os.path.splitext(wiz.p1.path)[0] + "_fit.poni"
        p, _ = QFileDialog.getSaveFileName(self, tr("save_title"), default,
                                           "poni (*.poni)")
        if not p:
            return
        if j is not None:
            # 多帧联合 poni(全局 D/rot+帧间中位中心;口径同批量导出)
            import numpy as _np
            p1c = float(_np.median(j["p1s"]))
            p2c = float(_np.median(j["p2s"]))
            px = j["px_um"] * 1e-6
            sig = j.get("sigma_boot")
            txt = (f"# Joint calibration ({j['nf']} frames, {j['ph']})\n"
                   f"# sigma_D = "
                   f"{('%.4f' % sig) if sig else 'n/a'} mm (bootstrap)\n"
                   f"PixelSize1: {px}\nPixelSize2: {px}\n"
                   f"Distance: {j['dist']}\n"
                   f"Poni1: {p1c}\nPoni2: {p2c}\n"
                   f"Rot1: {j['rot1']}\nRot2: {j['rot2']}\nRot3: 0.0\n"
                   f"Wavelength: {j['lam'] * 1e-10}\n")
            if wiz.p1.spline:
                # F4 口径:在线已校正图+poni 带 SplineFile ≡ 原图+几何
                txt += f"SplineFile: {wiz.p1.spline}\n"
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt)
            QMessageBox.information(self, tr("saved"),
                                    tr("saved_msg", p=p))
            return
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa
        ai = hybrid_ai(wiz.res, float(wiz.p2.e_px.text()))
        if wiz.p1.spline:
            # F4：poni 带 SplineFile，下游 pyFAI 按同一套畸变处理
            # （口径一致性由 _t5_sharp 实证：inv图+平几何 ≡ raw+spline几何）
            set_splinefile(ai, wiz.p1.spline)
        ai.save(p)
        QMessageBox.information(self, tr("saved"), tr("saved_msg", p=p))

    def _copy(self):
        QApplication.clipboard().setText(self.sum.toPlainText())
        QMessageBox.information(self, tr("copied"), tr("copied_msg"))


class Wizard(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("app_title"))
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage)
        self.res = None
        self.res_j = None      # 多帧联合结果(2026-11-08;None=单张会话)
        # 主题令牌（与专家版同一记忆；spec：颜色一律从令牌取）
        from PyQt5.QtCore import QSettings
        self.st = QSettings("pyfai", "gui_solver")
        self.theme_name = str(self.st.value("theme", "light"))
        self.p1 = PageImg()
        self.p2 = PageParam()
        self.p3 = PageSolve()
        self.p4 = PageCheck()
        self.p5 = PageDone()
        for p in (self.p1, self.p2, self.p3, self.p4, self.p5):
            self.addPage(p)
        nxt = self.button(QWizard.NextButton)
        nxt.setText(tr("btn_next"))
        fin = self.button(QWizard.FinishButton)
        fin.setText(tr("btn_finish"))
        # 🌐 语言菜单键（CustomButton1 落在按钮栏，显示当前语言母语名）
        self.setOption(QWizard.HaveCustomButton1, True)
        self.cb = self.button(QWizard.CustomButton1)
        self.cb.setText("🌐 " + LANG_NAMES[get_lang()])
        self.cb.clicked.connect(self._lang_menu)
        # 整窗拖放（spec §7）：拖到窗口任何位置都算数
        self.setAcceptDrops(True)
        # 记住窗口几何（spec §7）
        try:
            geo = self.st.value("wiz_geo")
            if geo:
                self.restoreGeometry(geo)
            else:
                self.resize(760, 620)
        except Exception:
            self.resize(760, 620)

    def _lang_menu(self):
        """🌐 弹六语菜单；选中 → 换语言 → 同逻辑重建窗口。"""
        from PyQt5.QtWidgets import QMenu
        m = QMenu(self)
        cur = get_lang()
        acts = {}
        for code in ("zh", "zh_TW", "en", "fr", "de", "ru"):
            a = m.addAction(("● " if code == cur else "  ") + LANG_NAMES[code])
            a.setCheckable(True)
            a.setChecked(code == cur)
            acts[a] = code
        chosen = m.exec_(self.cb.mapToGlobal(self.cb.rect().bottomLeft()))
        if chosen is None or acts[chosen] == cur:
            return
        set_lang(acts[chosen])
        self._rebuild()

    def _rebuild(self):
        """换语言后重建窗口：新窗先保活（防 GC 闪退），旧窗销毁。"""
        global _CUR
        self.st.setValue("wiz_geo", self.saveGeometry())
        new = Wizard()
        new.show()
        _CUR = new            # 先保活再删旧的，无空窗间隙
        self.deleteLater()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if not urls:
            return
        p = urls[0].toLocalFile()
        if self.currentPage() is self.p1:
            self.p1._load(p)
        else:
            QMessageBox.information(self, tr("switch_title"),
                                    tr("switch_msg"))

    def closeEvent(self, e):
        self.st.setValue("wiz_geo", self.saveGeometry())
        super().closeEvent(e)


def main():
    global _CUR
    if "--selftest" in sys.argv:      # 打包自检(2026-11-08):复用主窗口版
        from gui_solver import _selftest
        _selftest()
        return
    app = QApplication(sys.argv)

    # pythonw 无控制台：任何未捕获异常写日志，闪退有据可查
    def _hook(t, v, tb):
        import tempfile
        fp = os.path.join(tempfile.gettempdir(), "diffwizard_error.log")
        with open(fp, "a", encoding="utf-8") as f:
            f.write("".join(traceback.format_exception(t, v, tb)) + "\n")
    sys.excepthook = _hook

    import theme
    from PyQt5.QtCore import QSettings
    name = str(QSettings("pyfai", "gui_solver").value("theme", "light"))
    theme.apply_app(app, name)   # 与专家版同一记忆、同一令牌
    _CUR = Wizard()
    _CUR.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
