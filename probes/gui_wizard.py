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

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QFont
from PyQt5.QtWidgets import (
    QApplication, QWizard, QWizardPage, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox)

from gui_solver import (load_any, sniff_meta, SolveWorker, CheckWorker,
                        check_std, hybrid_ai, np, apply_spline,
                        set_splinefile)
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
            self._load(urls[0].toLocalFile())

    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("p1_pick"), "",
                                           IMG_FILTER)
        if p:
            self._load(p)

    def _load(self, p):
        try:
            self.img = load_any(p)
        except Exception as e:
            QMessageBox.warning(self, tr("p1_openfail"), str(e))
            return
        self.path = p
        self.img_raw = None          # 新图=全新会话，样条状态清零（F6）
        self.spline = None
        self.info.setText(tr("p1_loaded", name=os.path.basename(p),
                             w=self.img.shape[1], h=self.img.shape[0]))
        self._sp_refresh()
        self.completeChanged.emit()

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
        g.addWidget(QLabel(tr("p2_lam_hint")), 1, 2)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        g.addWidget(self.note, 2, 0, 3, 1)
        self.setLayout(g)

    def initializePage(self):
        wiz = self.wizard()
        meta = sniff_meta(wiz.p1.path)
        if "px" in meta:
            self.e_px.setText(f"{meta['px']:g}")
        else:
            self.e_px.setText("172")
        self.e_lam.setText(f"{meta['lam']:g}" if "lam" in meta else "")
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

    def validatePage(self):
        try:
            v = float(self.e_px.text())
            assert v > 1
            return True
        except Exception:
            QMessageBox.warning(self, tr("p2_badpx_title"), tr("p2_badpx"))
            return False


class PageSolve(QWizardPage):
    """③自动求解：进入页面即开跑，完成才能下一步。"""

    def __init__(self):
        super().__init__()
        self.setTitle(tr("p3_title"))
        self.setSubTitle(tr("p3_sub"))
        self._ok = False
        self.worker = None
        lay = QVBoxLayout()
        self.pb = QProgressBar(); self.pb.setRange(0, 0)
        self.lab = QLabel(tr("p3_ready"))
        self.lab.setWordWrap(True)
        self.out = QTextEdit(); self.out.setReadOnly(True)
        lay.addWidget(self.pb)
        lay.addWidget(self.lab)
        lay.addWidget(self.out, 1)
        self.setLayout(lay)

    def initializePage(self):
        if self.worker is not None and self.worker.isRunning():
            return
        wiz = self.wizard()
        self._ok = False
        self.pb.setVisible(True)
        lam_t = wiz.p2.e_lam.text().strip()
        self.worker = SolveWorker(wiz.p1.img,
                                  float(lam_t) if lam_t else None,
                                  float(wiz.p2.e_px.text()))
        self.worker.done.connect(self._done)
        self.worker.fail.connect(self._fail)
        self.lab.setText(tr("p3_running"))
        self.worker.start()

    def _fail(self, tb):
        self.pb.setVisible(False)
        self.lab.setText(tr("p3_fail"))
        self.out.setPlainText(tb)

    def _done(self, res):
        self.pb.setVisible(False)
        wiz = self.wizard()
        wiz.res = res
        if res is None:
            self.lab.setText(tr("p3_none"))
            self.out.setPlainText(tr("p3_honest"))
            self.completeChanged.emit()
            return
        px_um = float(wiz.p2.e_px.text())
        D_mm = res["D_px"] * px_um * 1e-3
        tf = res.get("tf")
        tilt = (tr("s_tilt_v", a=tf["rot1"] * 1e3, b=tf["rot2"] * 1e3)
                if tf else tr("s_tilt_none"))
        self.out.setPlainText(
            "\n".join([
                tr("s_phase", ph=res["ph"]),
                tr("s_lam", v=f"{res['lam']:.4f}"),
                tr("s_dist", v=f"{D_mm:.2f}"),
                tr("s_center", x=f"{res['cx']:.1f}", y=f"{res['cy']:.1f}"),
                tr("s_match", m=res["sol"][3], rms=f"{res['sol'][4]:.2f}",
                   ps=f"{res['sol'][6]:.3f}"),
                tr("s_tilt", v=tilt)]))
        warn = "" if wiz.p2.e_lam.text().strip() else tr("p3_lamwarn")
        self.lab.setText(tr("p3_done") + warn)
        self._ok = res is not None
        self.completeChanged.emit()

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

    def _done(self, c):
        self.out.setPlainText("\n".join(c["lines"]))
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
        p1 = wiz.p1
        p1.path = None
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
        default = os.path.splitext(wiz.p1.path)[0] + "_fit.poni"
        p, _ = QFileDialog.getSaveFileName(self, tr("save_title"), default,
                                           "poni (*.poni)")
        if not p:
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
