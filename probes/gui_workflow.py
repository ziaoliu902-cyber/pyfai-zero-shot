# -*- coding: utf-8 -*-
"""gui_workflow.py：流程图式交互标定工作台。

将衍射几何盲解封装为 5 个可视化流程图节点：
  ① 数据导入 → ② 掩膜与畸变 → ③ 自动标定 → ④ Cake 体检 → ⑤ 导出结果
点击任意节点可查看/配置该步参数与输出摘要，也可点击【一键运行全流程】自动贯通执行。

界面文案经 i18n_wf 词条表本地化（11 种语言，菜单栏「语言/Language」实时切换并重翻译；
语言选择持久化到 QSettings，首次启动按系统区域自动挑选）。
"""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(ROOT)


def _data_root():
    """定位 data 目录（兼容源码运行与 PyInstaller 打包后的 _MEIPASS / _internal）。"""
    cands = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            cands.append(os.path.join(meipass, "data"))
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        cands.append(os.path.join(exe_dir, "data"))
        cands.append(os.path.join(exe_dir, "_internal", "data"))
    cands.append(os.path.join(PROJ_ROOT, "data"))
    for d in cands:
        if os.path.isdir(d):
            return d
    return os.path.join(PROJ_ROOT, "data")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import matplotlib
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if "Microsoft YaHei" not in matplotlib.rcParams.get("font.sans-serif", []):
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
matplotlib.rcParams["axes.unicode_minus"] = False

import theme
from i18n_wf import LANG_NAMES, get_lang, init_lang, is_rtl, localize_conf, set_lang, tr
from auto_mask import build_auto_mask, detect_beamstop_shadow
from cake_report import (
    _geom_from_res,
    _mask_overlay,
    compute_cake,
    draw_cake,
    export_diagnostic_figure,
    score_cake_profile,
)
from confidence_map import evaluate_confidence
from custom_phase import load_and_register_phase
from gui_solver import load_any, load_mask_file, sniff_meta
from multiframe_calib import calibrate_multiframe
from poni_out import _lam_A, fit_tilt, make_pyfai_fwd, write_poni
from solver_core import _r_model, auto_offset, ring_radii, solve_image, to_canvas
from torchfai.phases import PHASES, PHASE_NAMES, phase_qs

# 界面文案一律经 i18n_wf.tr 取词（11 语言实时切换）；含通配符的文件过滤器、
# 流程节点名等也在调用点取 tr，而非模块级常量——否则冻结在 import 时刻的语言。
STAGE_KEYS = [
    ("st1_t", "st1_s"), ("st2_t", "st2_s"), ("st3_t", "st3_s"),
    ("st4_t", "st4_s"), ("st5_t", "st5_s"),
]


def _stages():
    """按当前语言返回 5 个流程节点的 (标题, 默认副标题)。"""
    return [(tr(t), tr(s)) for t, s in STAGE_KEYS]


STATE_STYLE = {
    # 状态 → (徽标词条键, 前景色, 背景色)；徽标文字在渲染时经 tr 本地化
    "idle": ("state_idle", "#9CA3AF", "#F3F4F6"),
    "running": ("state_running", "#2563EB", "#DBEAFE"),
    "done": ("state_done", "#0A7D0A", "#DCFCE7"),
    "skipped": ("state_skipped", "#6B7280", "#E5E7EB"),
    "error": ("state_error", "#C2410C", "#FFEDD5"),
}


class FlowArrow(QWidget):
    """节点间的水平连接箭头。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 56)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#9CA3AF"), 2.0)
        p.setPen(pen)
        p.setBrush(QColor("#9CA3AF"))
        ym = self.height() // 2
        x1, x2 = 2, self.width() - 4
        p.drawLine(x1, ym, x2 - 4, ym)
        head = QPolygonF([
            QPointF(x2, ym),
            QPointF(x2 - 7, ym - 4.5),
            QPointF(x2 - 7, ym + 4.5),
        ])
        p.drawPolygon(head)


class FlowNodeCard(QFrame):
    """可点击的流程图节点卡片，展示步骤名、状态徽标与单行摘要。"""

    clicked = pyqtSignal(int)

    def __init__(self, idx, title, default_sub, parent=None):
        super().__init__(parent)
        self.idx = int(idx)
        self.state = "idle"
        self.selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(148)
        self.setFixedHeight(64)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(6)
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 9.8pt;")
        self.lbl_badge = QLabel("")  # 文字由 _apply_style 经 tr 填充
        self.lbl_badge.setAlignment(Qt.AlignCenter)
        top.addWidget(self.lbl_title, 1)
        top.addWidget(self.lbl_badge, 0)

        self.lbl_sub = QLabel(default_sub)
        self.lbl_sub.setStyleSheet("font-size: 8.4pt; color: #6B7280;")
        lay.addLayout(top)
        lay.addWidget(self.lbl_sub)
        self._apply_style()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit(self.idx)
        super().mousePressEvent(ev)

    def set_selected(self, sel):
        self.selected = bool(sel)
        self._apply_style()

    def set_texts(self, title, sub):
        """语言切换后由窗口统一重设标题/副标题。"""
        self.lbl_title.setText(title)
        self.lbl_sub.setText(sub)

    def set_status(self, state, summary=None):
        self.state = state if state in STATE_STYLE else "idle"
        if summary is not None:
            self.lbl_sub.setText(str(summary))
        self._apply_style()

    def _apply_style(self):
        t = theme.TOKENS["light"]
        label_key, fg, bg = STATE_STYLE[self.state]
        label = tr(label_key)
        self.lbl_badge.setText(label)
        self.lbl_badge.setStyleSheet(
            f"color: {fg}; background: {bg}; border-radius: 9px; "
            f"padding: 1px 7px; font-size: 8pt; font-weight: 600;"
        )
        border_c = t["accent"] if self.selected else (fg if self.state == "done" else t["border"])
        border_w = 2 if self.selected else 1
        card_bg = "#EFF6FF" if self.selected else t["surface"]
        self.setStyleSheet(
            f"FlowNodeCard {{ background: {card_bg}; border: {border_w}px solid {border_c}; "
            f"border-radius: 8px; }}"
        )


class WorkflowWorker(QThread):
    """后台流水线执行器：掩膜构建 → 单帧/多帧标定 → Cake 展开与评分。"""

    stage_changed = pyqtSignal(int, str, str)
    log_line = pyqtSignal(str)
    finished_all = pyqtSignal(object)
    failed = pyqtSignal(int, str)

    def __init__(self, paths, px1_um, px2_um, lam_A, phase_opt=None,
                 auto_gaps=True, auto_bad=True, auto_bs=True,
                 ext_mask=None, spline_path=None, d_ref_mm=None):
        super().__init__()
        self.paths = list(paths)
        self.px1_um = float(px1_um)
        self.px2_um = float(px2_um) if px2_um else float(px1_um)
        self.lam_A = float(lam_A)
        self.phase_opt = list(phase_opt) if phase_opt else None
        self.auto_gaps = bool(auto_gaps)
        self.auto_bad = bool(auto_bad)
        self.auto_bs = bool(auto_bs)
        self.ext_mask = ext_mask
        self.spline_path = spline_path or None
        self.d_ref_mm = d_ref_mm

    def run(self):
        cur_stage = 1
        try:
            px1_m = self.px1_um * 1e-6
            px2_m = self.px2_um * 1e-6
            is_multi = len(self.paths) >= 2

            # ---- Stage 2: 掩膜与畸变 ----
            cur_stage = 1
            self.stage_changed.emit(1, "running", tr("sum_mask_run"))
            self.log_line.emit(tr("log2_prepare", n=len(self.paths)))
            first_raw = auto_offset(load_any(self.paths[0]))
            mask = None
            if self.ext_mask is not None and self.ext_mask.shape == first_raw.shape:
                mask = self.ext_mask.astype(bool).copy()
            if self.auto_gaps or self.auto_bad or self.auto_bs:
                m_auto, mst = build_auto_mask(
                    first_raw,
                    include_gaps=self.auto_gaps,
                    include_bad=self.auto_bad,
                    include_beamstop=self.auto_bs,
                )
                mask = m_auto if mask is None else (mask | m_auto)
                self.log_line.emit(tr(
                    "log2_auto", g=mst["n_gap"], b=mst["n_bad"], s=mst["n_beamstop"],
                    t=mst["n_total"], p=f"{mst['frac']*100:.1f}"))
            else:
                self.log_line.emit(tr("log2_off"))
            sp_tag = f" + {os.path.basename(self.spline_path)}" if self.spline_path else ""
            # 无掩膜时以 "0 px" 表述（语言中立），避免为「无」单独造词
            mask_tag = f"{int(mask.sum())} px" if mask is not None and mask.any() else "0 px"
            self.stage_changed.emit(1, "done", tr("sum_mask_done", m=mask_tag, sp=sp_tag))

            # ---- Stage 3: 自动标定 ----
            cur_stage = 2
            self.stage_changed.emit(2, "running", tr("sum_solve_run"))
            if is_multi:
                self.log_line.emit(tr("log3_multi_start", n=len(self.paths)))
                mf = calibrate_multiframe(
                    self.paths, px1_m, lam=self.lam_A, px2_m=px2_m,
                    phases=self.phase_opt, spline=self.spline_path, n_workers=1,
                )
                if not mf.get("ok"):
                    raise RuntimeError(str(mf.get("error") or tr("err_multifail")))
                kept_imgs = [auto_offset(load_any(p)) for p in mf["kept_files"]]
                img_work = np.mean(kept_imgs, axis=0).astype(np.float64)
                if self.auto_gaps or self.auto_bad or self.auto_bs or self.ext_mask is not None:
                    m2, _ = build_auto_mask(
                        img_work,
                        include_gaps=self.auto_gaps,
                        include_bad=self.auto_bad,
                        include_beamstop=self.auto_bs,
                    )
                    if self.ext_mask is not None and self.ext_mask.shape == img_work.shape:
                        m2 = m2 | self.ext_mask.astype(bool)
                    mask = m2

                ph = mf["phase"]
                lam_u = float(mf["lam_A"])
                dist_m = float(mf["dist_mm"]) * 1e-3
                poni1_m = float(mf["poni1_m"])
                poni2_m = float(mf["poni2_m"])
                rot1 = float(mf["rot1"])
                rot2 = float(mf["rot2"])

                canvas, s, _, _ = to_canvas(img_work)
                res_mean = solve_image(
                    img_work, px1_m, lam=lam_u, phases=[ph], px2_m=px2_m,
                    mask=mask, tilt_refine=False, d_ref_mm=self.d_ref_mm,
                )
                if res_mean is not None:
                    res = res_mean
                else:
                    px_c = 0.5 * (px1_m + px2_m) * float(s)
                    cx_c = (poni2_m / px2_m - 0.5 + 0.5) / float(s) - 0.5
                    cy_c = (poni1_m / px1_m - 0.5 + 0.5) / float(s) - 0.5
                    res = dict(
                        phase=ph,
                        sol=(cx_c, cy_c, dist_m / px_c, 6, 1.5, 0.0, 0.45),
                        lam=lam_u, canvas=canvas, s=s, px_m=px_c,
                        m=6, rms=1.5, ps=0.45, az_coverage=1.0, native_2d=False,
                    )
                res["tilt_ref"] = dict(
                    applied=True, dist=dist_m, poni1=poni1_m, poni2=poni2_m,
                    rot1=rot1, rot2=rot2, tilt_deg=float(mf["tilt_deg"]),
                )
                self.log_line.emit(tr(
                    "log3_multi_done", k=mf["n_kept"], n=mf["n_total"], d=mf["n_dark"],
                    o=mf["n_outliers"], ph=ph, dmm=f"{mf['dist_mm']:.3f}",
                    r1=f"{rot1*1e3:+.2f}", r2=f"{rot2*1e3:+.2f}"))
            else:
                img_work = first_raw
                self.log_line.emit(tr("log3_single_start"))
                res = solve_image(
                    img_work, px1_m, lam=self.lam_A, phases=self.phase_opt,
                    px2_m=px2_m, mask=mask, d_ref_mm=self.d_ref_mm,
                )
                if not res:
                    raise RuntimeError(tr("err_solvefail"))
                dist_m, poni1_m, poni2_m, rot1, rot2 = _geom_from_res(res, px1_m, px2_m)
                ph = res["phase"]
                lam_u = float(res["lam"])
                if self.spline_path and ph:
                    qs_sp = phase_qs(ph)
                    if len(qs_sp) >= 4:
                        canvas = res["canvas"]
                        s = float(res["s"])
                        sol = res["sol"]
                        fwd_sp = make_pyfai_fwd(px1_m, px2_m, self.spline_path)
                        tf_sp = fit_tilt(
                            canvas, float(sol[0]), float(sol[1]), float(sol[2]),
                            qs_sp, lam_u, px1_m * s, px2_m * s, sfull=int(round(s)),
                            fwd=fwd_sp, nsec=36, win=6.0, mass_rel=0.10, bump=1.5,
                            suspect_tol=5e-3, min_gap=14.0,
                        )
                        if tf_sp is not None and tf_sp["nrings"] >= 4 and tf_sp["rms5"] < 0.006:
                            if float(tf_sp.get("imp", 0.0)) >= 0.15:
                                dist_m = float(tf_sp["dist"])
                                poni1_m = float(tf_sp["poni1"])
                                poni2_m = float(tf_sp["poni2"])
                                rot1 = float(tf_sp["rot1"])
                                rot2 = float(tf_sp["rot2"])
                            elif "dist3" in tf_sp:
                                dist_m = float(tf_sp["dist3"])
                                poni1_m = float(tf_sp["poni1_3"])
                                poni2_m = float(tf_sp["poni2_3"])
                                rot1 = 0.0
                                rot2 = 0.0
                            res["tilt_ref"] = dict(
                                applied=True, dist=dist_m, poni1=poni1_m, poni2=poni2_m,
                                rot1=rot1, rot2=rot2, tilt_deg=float(np.hypot(rot1, rot2) * 180 / np.pi),
                            )
                self.log_line.emit(tr(
                    "log3_single_done", ph=ph, d=f"{dist_m*1000:.3f}", m=res["m"],
                    rms=f"{res['rms']:.2f}", r1=f"{rot1*1e3:+.2f}", r2=f"{rot2*1e3:+.2f}"))
            self.stage_changed.emit(2, "done", tr("sum_geom_done", ph=ph, d=f"{dist_m*1000:.2f}"))

            # ---- 二段式离轴束挡掩膜（步骤 3 解出真实束心后补抓角落束挡阴影） ----
            cx_nat = poni2_m / px2_m - 0.5
            cy_nat = poni1_m / px1_m - 0.5
            H_w, W_w = img_work.shape[:2]
            if self.auto_bs and np.hypot(cx_nat - W_w / 2.0, cy_nat - H_w / 2.0) > 0.08 * min(H_w, W_w):
                m_bs_off = detect_beamstop_shadow(img_work, cx=cx_nat, cy=cy_nat)
                if m_bs_off.any():
                    mask = m_bs_off if mask is None else (mask | m_bs_off)
                    self.log_line.emit(tr("log3_bs2", n=int(m_bs_off.sum())))
                    sp_tag = f" + {os.path.basename(self.spline_path)}" if self.spline_path else ""
                    self.stage_changed.emit(1, "done", tr("sum_mask_done", m=f"{int(mask.sum())} px", sp=sp_tag))

            # ---- Stage 4: Cake 体检 ----
            cur_stage = 3
            self.stage_changed.emit(3, "running", tr("sum_cake_run"))
            self.log_line.emit(tr("log4_start"))
            cake = compute_cake(
                img_work, dist_m, poni1_m, poni2_m, rot1=rot1, rot2=rot2,
                px1=px1_m, px2=px2_m, lam_A=lam_u, mask=mask, spline=self.spline_path,
            )
            q_1d = cake["q_axis"]
            prof_1d = np.nanmean(cake["cake"], axis=0)
            qs = phase_qs(ph) if ph else np.array([], dtype=float)
            i_rel = PHASES[ph][1] if ph and ph in PHASES else None
            sc = score_cake_profile(q_1d, prof_1d, qs, I_rel=i_rel)

            m_v = int(res.get("m", res["sol"][3]))
            ps_v = float(res.get("ps", res["sol"][6]))
            mf_v = res.get("mf_score")
            ps_eff = ps_v
            if mf_v is not None and np.isfinite(mf_v) and m_v >= 4:
                ps_eff = max(ps_eff, float(mf_v))
            if sc["n_peak_hit"] >= 3 and np.isfinite(sc["ps_refined"]):
                ps_eff = max(ps_eff, float(sc["ps_refined"]))

            cv_res = res.get("cv")
            cv_ratio = cv_res.get("cv_ratio") if isinstance(cv_res, dict) else None
            conf = evaluate_confidence(
                m=m_v, ps=ps_eff,
                az_cov=float(res.get("az_coverage", 1.0)),
                cv_ratio=cv_ratio,
                has_d_ref=bool(res.get("d_ref_ok", False)),
            )
            res["confidence"] = conf
            dq_txt = f"{sc['dq_med']:.4f} Å⁻¹" if np.isfinite(sc["dq_med"]) else "N/A"
            # 置信度 tier/badge/说明在发射前先经 localize_conf 转成当前语言
            loc = localize_conf(conf, has_d_ref=bool(res.get("d_ref_ok", False)))
            self.log_line.emit(tr(
                "log4_done", p=f"{sc['ps_refined']:.3f}", dq=dq_txt, n=sc["n_peak_hit"],
                tier=loc["tier"], badge=loc["badge"]))
            self.stage_changed.emit(3, "done", tr(
                "sum_cake_done", tier=loc["tier"], p=f"{sc['ps_refined']:.2f}"))
            self.stage_changed.emit(4, "idle", tr("sum_export_idle"))

            payload = dict(
                img=img_work,
                mask=mask,
                spline=self.spline_path,
                res=res,
                cake=cake,
                score=sc,
                conf=conf,
                geom=(dist_m, poni1_m, poni2_m, rot1, rot2),
                px1_m=px1_m,
                px2_m=px2_m,
                lam_A=lam_u,
                is_multi=is_multi,
            )
            self.finished_all.emit(payload)
        except Exception:
            self.failed.emit(cur_stage, traceback.format_exc(limit=6))


class WorkflowWindow(QMainWindow):
    """流程图式 X 射线衍射几何标定工作台主窗口。"""

    HC_KEV_A = 12.39841984

    def __init__(self):
        super().__init__()
        # 语言初始化：QSettings 记忆 > 系统区域；阿拉伯语等 RTL 布局镜像
        cur_lang = init_lang()
        self.setWindowTitle(tr("app_title"))
        QApplication.setLayoutDirection(Qt.RightToLeft if is_rtl() else Qt.LeftToRight)
        self.resize(1340, 820)
        self.paths = []
        self.img_preview = None
        self.ext_mask = None
        self.preview_mask = None
        self.spline_path = None
        self.payload = None
        self.worker = None
        # 语言切换重翻译所需的动态状态快照（_retranslate 据此再生成动态文案）
        self._card_states = ["idle"] * 5
        self._n_files = 0
        self._first_name = ""
        self._meta_src = None          # 首帧元数据来源串（None → meta_none）
        self._qa_state = "wait"        # wait / reset / full
        self._exp_last = None          # ("poni"|"fig", 绝对路径)
        self._ext_mask_name = None
        self._build_lang_menu(cur_lang)

        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setContentsMargins(10, 10, 10, 8)
        root_lay.setSpacing(8)

        # ---- 顶部交互式流程图工具条 ----
        self.top_box = QGroupBox(tr("flow_box"))
        top_lay = QHBoxLayout(self.top_box)
        top_lay.setContentsMargins(10, 12, 10, 10)
        top_lay.setSpacing(6)

        self.cards = []
        for idx, (title, sub) in enumerate(_stages()):
            if idx > 0:
                top_lay.addWidget(FlowArrow())
            card = FlowNodeCard(idx, title, sub)
            card.clicked.connect(self.select_stage)
            self.cards.append(card)
            top_lay.addWidget(card, 1)

        ctrl_col = QVBoxLayout()
        ctrl_col.setSpacing(6)
        self.b_run_all = QPushButton(tr("run_all"))
        self.b_run_all.setProperty("primary", True)
        self.b_run_all.setMinimumHeight(34)
        self.b_run_all.setEnabled(False)
        self.b_run_all.clicked.connect(self.run_all)
        self.b_reset = QPushButton(tr("reset"))
        self.b_reset.clicked.connect(self.reset_workflow)
        ctrl_col.addWidget(self.b_run_all)
        ctrl_col.addWidget(self.b_reset)
        top_lay.addLayout(ctrl_col)
        root_lay.addWidget(self.top_box)

        self.pb = QProgressBar()
        self.pb.setRange(0, 0)
        self.pb.setVisible(False)
        self.pb.setFixedHeight(8)
        root_lay.addWidget(self.pb)

        # ---- 主工作区：左侧阶段面板 + 右侧可视化视图 ----
        splitter = QSplitter(Qt.Horizontal)

        left_wrap = QWidget()
        left_lay = QVBoxLayout(left_wrap)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_wrap.setMinimumWidth(360)
        left_wrap.setMaximumWidth(450)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_stage0())
        self.stack.addWidget(self._build_stage1())
        self.stack.addWidget(self._build_stage2())
        self.stack.addWidget(self._build_stage3())
        self.stack.addWidget(self._build_stage4())
        left_lay.addWidget(self.stack)

        # 右侧可视化多页签
        self.tabs = QTabWidget()
        self.fig_2d, self.ax_2d, self.cv_2d, tab_2d = self._make_plot_tab()
        self.fig_cake, self.ax_cake, self.cv_cake, tab_cake = self._make_plot_tab()
        self.fig_1d, self.ax_1d, self.cv_1d, tab_1d = self._make_plot_tab()
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)

        self.tabs.addTab(tab_2d, tr("tab_2d"))
        self.tabs.addTab(tab_cake, tr("tab_cake"))
        self.tabs.addTab(tab_1d, tr("tab_1d"))
        self.tabs.addTab(self.txt_log, tr("tab_log"))

        splitter.addWidget(left_wrap)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_lay.addWidget(splitter, 1)

        self.status_lbl = QLabel(tr("status_idle"))
        self.status_lbl.setProperty("muted", True)
        root_lay.addWidget(self.status_lbl)

        self.select_stage(0)

    def _make_plot_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        fig = Figure(figsize=(6.4, 5.2), dpi=100)
        ax = fig.add_subplot(111)
        cv = FigureCanvasQTAgg(fig)
        tb = NavigationToolbar2QT(cv, w)
        lay.addWidget(tb)
        lay.addWidget(cv, 1)
        return fig, ax, cv, w

    # ---- 构建 5 个阶段的配置面板 ----
    def _build_stage0(self):
        self.gb0 = QGroupBox(tr("s0_title"))
        lay = QVBoxLayout(self.gb0)

        self.drop_zone = QLabel(tr("s0_drop"))
        self.drop_zone.setAlignment(Qt.AlignCenter)
        self.drop_zone.setMinimumHeight(130)
        self.drop_zone.setAcceptDrops(True)
        self.drop_zone.dragEnterEvent = self._on_drag_enter
        self.drop_zone.dropEvent = self._on_drop
        t = theme.TOKENS["light"]
        self.drop_zone.setStyleSheet(
            f"border: 2px dashed {t['border']}; border-radius: 8px; "
            f"background: {t['bg']}; color: {t['muted']}; padding: 10px;"
        )
        lay.addWidget(self.drop_zone)

        self.b_pick0 = QPushButton(tr("s0_pick"))
        self.b_pick0.setProperty("primary", True)
        self.b_pick0.clicked.connect(self.pick_images)
        lay.addWidget(self.b_pick0)

        self.lbl_file_info = QLabel(tr("s0_nofile"))
        self.lbl_file_info.setWordWrap(True)
        self.lbl_file_info.setProperty("muted", True)
        lay.addWidget(self.lbl_file_info)

        self.demo_box = QGroupBox(tr("s0_demo"))
        dl = QVBoxLayout(self.demo_box)
        self.b_demo_ceria = QPushButton(tr("s0_demo_ceria"))
        self.b_demo_ceria.clicked.connect(self.load_demo_ceria)
        self.b_demo_alumina = QPushButton(tr("s0_demo_alumina"))
        self.b_demo_alumina.clicked.connect(self.load_demo_alumina)
        dl.addWidget(self.b_demo_ceria)
        dl.addWidget(self.b_demo_alumina)
        lay.addWidget(self.demo_box)

        lay.addStretch(1)
        self.b_next0 = QPushButton(tr("s0_next"))
        self.b_next0.clicked.connect(lambda: self.select_stage(1))
        lay.addWidget(self.b_next0)
        return self.gb0

    def _build_stage1(self):
        self.gb1 = QGroupBox(tr("s1_title"))
        lay = QVBoxLayout(self.gb1)

        self.chk_gaps = QCheckBox(tr("s1_gaps"))
        self.chk_gaps.setChecked(True)
        self.chk_bad = QCheckBox(tr("s1_bad"))
        self.chk_bad.setChecked(True)
        self.chk_bs = QCheckBox(tr("s1_bs"))
        self.chk_bs.setChecked(True)
        lay.addWidget(self.chk_gaps)
        lay.addWidget(self.chk_bad)
        lay.addWidget(self.chk_bs)

        row_m = QHBoxLayout()
        self.b_extm = QPushButton(tr("s1_extm"))
        self.b_extm.clicked.connect(self.pick_ext_mask)
        self.b_clrm = QPushButton(tr("s1_clear"))
        self.b_clrm.clicked.connect(self.clear_ext_mask)
        row_m.addWidget(self.b_extm, 1)
        row_m.addWidget(self.b_clrm)
        lay.addLayout(row_m)
        self.lbl_ext_mask = QLabel(tr("s1_extm_none"))
        self.lbl_ext_mask.setProperty("muted", True)
        lay.addWidget(self.lbl_ext_mask)

        self.b_prev_mask = QPushButton(tr("s1_preview"))
        self.b_prev_mask.clicked.connect(self.preview_auto_mask)
        lay.addWidget(self.b_prev_mask)

        self.sp_box = QGroupBox(tr("s1_spbox"))
        sl = QVBoxLayout(self.sp_box)
        row_sp = QHBoxLayout()
        self.b_sp = QPushButton(tr("s1_sp_pick"))
        self.b_sp.clicked.connect(self.pick_spline)
        self.b_sp_clr = QPushButton(tr("s1_sp_off"))
        self.b_sp_clr.clicked.connect(self.clear_spline)
        row_sp.addWidget(self.b_sp, 1)
        row_sp.addWidget(self.b_sp_clr)
        sl.addLayout(row_sp)
        self.lbl_spline = QLabel(tr("s1_sp_none"))
        self.lbl_spline.setWordWrap(True)
        self.lbl_spline.setProperty("muted", True)
        sl.addWidget(self.lbl_spline)
        lay.addWidget(self.sp_box)

        lay.addStretch(1)
        self.b_next1 = QPushButton(tr("s1_next"))
        self.b_next1.clicked.connect(lambda: self.select_stage(2))
        lay.addWidget(self.b_next1)
        return self.gb1

    def _build_stage2(self):
        self.gb2 = QGroupBox(tr("s2_title"))
        lay = QVBoxLayout(self.gb2)

        grid = QGridLayout()
        self.lbl_px1 = QLabel(tr("s2_px1"))
        grid.addWidget(self.lbl_px1, 0, 0)
        self.e_px1 = QLineEdit("172")
        grid.addWidget(self.e_px1, 0, 1)

        self.lbl_px2 = QLabel(tr("s2_px2"))
        grid.addWidget(self.lbl_px2, 1, 0)
        self.e_px2 = QLineEdit("")
        self.e_px2.setPlaceholderText(tr("s2_px2_ph"))
        grid.addWidget(self.e_px2, 1, 1)

        self.lbl_lam = QLabel(tr("s2_lam"))
        grid.addWidget(self.lbl_lam, 2, 0)
        row_lam = QHBoxLayout()
        self.e_lam = QLineEdit("")
        self.e_lam.setPlaceholderText(tr("s2_lam_ph"))
        self.cb_lam_unit = QComboBox()
        self.cb_lam_unit.addItems(["Å", "keV"])
        row_lam.addWidget(self.e_lam, 1)
        row_lam.addWidget(self.cb_lam_unit)
        grid.addLayout(row_lam, 2, 1)

        self.lbl_phase = QLabel(tr("s2_phase"))
        grid.addWidget(self.lbl_phase, 3, 0)
        self.cb_phase = QComboBox()
        self._refresh_phase_combo()
        grid.addWidget(self.cb_phase, 3, 1)

        self.lbl_dref = QLabel(tr("s2_dref"))
        grid.addWidget(self.lbl_dref, 4, 0)
        self.e_dref = QLineEdit("")
        self.e_dref.setPlaceholderText(tr("s2_dref_ph"))
        grid.addWidget(self.e_dref, 4, 1)
        lay.addLayout(grid)

        self.b_imp = QPushButton(tr("s2_imp"))
        self.b_imp.clicked.connect(self.import_custom_phase)
        lay.addWidget(self.b_imp)

        self.lbl_calib_mode = QLabel(tr("s2_mode_wait"))
        self.lbl_calib_mode.setWordWrap(True)
        self.lbl_calib_mode.setProperty("muted", True)
        lay.addWidget(self.lbl_calib_mode)

        lay.addStretch(1)
        self.b_go = QPushButton(tr("s2_go"))
        self.b_go.setProperty("primary", True)
        self.b_go.clicked.connect(self.run_all)
        lay.addWidget(self.b_go)
        return self.gb2

    def _build_stage3(self):
        self.gb3 = QGroupBox(tr("s3_title"))
        lay = QVBoxLayout(self.gb3)

        self.txt_qa = QPlainTextEdit()
        self.txt_qa.setReadOnly(True)
        self.txt_qa.setPlainText(tr("s3_wait"))
        lay.addWidget(self.txt_qa, 1)

        row_t = QHBoxLayout()
        self.b_t2d = QPushButton(tr("s3_b2d"))
        self.b_t2d.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        self.b_tcake = QPushButton(tr("s3_bcake"))
        self.b_tcake.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        self.b_t1d = QPushButton(tr("s3_b1d"))
        self.b_t1d.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        row_t.addWidget(self.b_t2d)
        row_t.addWidget(self.b_tcake)
        row_t.addWidget(self.b_t1d)
        lay.addLayout(row_t)

        self.b_next3 = QPushButton(tr("s3_next"))
        self.b_next3.clicked.connect(lambda: self.select_stage(4))
        lay.addWidget(self.b_next3)
        return self.gb3

    def _build_stage4(self):
        self.gb4 = QGroupBox(tr("s4_title"))
        lay = QVBoxLayout(self.gb4)

        self.b_exp_poni = QPushButton(tr("s4_poni"))
        self.b_exp_poni.setProperty("primary", True)
        self.b_exp_poni.setEnabled(False)
        self.b_exp_poni.clicked.connect(self.export_poni)

        self.b_exp_fig = QPushButton(tr("s4_fig"))
        self.b_exp_fig.setEnabled(False)
        self.b_exp_fig.clicked.connect(self.export_diag_fig)

        self.b_copy_txt = QPushButton(tr("s4_copy"))
        self.b_copy_txt.setEnabled(False)
        self.b_copy_txt.clicked.connect(self.copy_summary)

        lay.addWidget(self.b_exp_poni)
        lay.addWidget(self.b_exp_fig)
        lay.addWidget(self.b_copy_txt)

        self.lbl_export = QLabel(tr("s4_hint"))
        self.lbl_export.setWordWrap(True)
        self.lbl_export.setProperty("muted", True)
        lay.addWidget(self.lbl_export)
        lay.addStretch(1)
        return self.gb4

    # ---- 交互逻辑 ----
    def select_stage(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, c in enumerate(self.cards):
            c.set_selected(i == idx)

    # ---- 多语言：菜单构建 / 切换 / 全量重翻译 ----
    def _build_lang_menu(self, cur_lang):
        """菜单栏「语言/Language」：母语名常显，互斥单选。"""
        self.menu_lang = self.menuBar().addMenu(tr("menu_lang"))
        grp = QActionGroup(self)
        self._lang_acts = []
        for code, name in LANG_NAMES.items():
            act = QAction(name, self)
            act.setCheckable(True)
            act.setData(code)
            act.setChecked(code == cur_lang)
            act.triggered.connect(lambda _=False, c=code: self.switch_lang(c))
            grp.addAction(act)
            self.menu_lang.addAction(act)
            self._lang_acts.append(act)

    def switch_lang(self, code):
        """切换语言并全量重翻译界面；语言选择持久化，重启后沿用。"""
        if code == get_lang():
            return
        set_lang(code)
        QApplication.setLayoutDirection(Qt.RightToLeft if is_rtl() else Qt.LeftToRight)
        self._apply_mpl_font()
        self._retranslate()
        self.status_lbl.setText(tr("lang_changed", n=LANG_NAMES.get(code, code)))

    def _apply_mpl_font(self):
        """RTL 语种下给 matplotlib 优先安排含阿拉伯字形的西文字体。

        说明：matplotlib 不做双向文本整形，图内阿拉伯文可能呈非连写形态，
        属渲染层已知限制；Qt 部件（页签/按钮/日志）不受影响。
        """
        base = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
        if is_rtl():
            matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "Tahoma", "Arial"] + base
        else:
            matplotlib.rcParams["font.sans-serif"] = base
        matplotlib.rcParams["axes.unicode_minus"] = False

    def _retranslate(self):
        """按当前语言重设全部静态控件文本，并依据状态快照再生成动态文案。"""
        self.setWindowTitle(tr("app_title"))
        self.menu_lang.setTitle(tr("menu_lang"))
        for act in self._lang_acts:
            act.setChecked(act.data() == get_lang())
        self.top_box.setTitle(tr("flow_box"))
        self.b_run_all.setText(tr("run_all"))
        self.b_reset.setText(tr("reset"))
        self.tabs.setTabText(0, tr("tab_2d"))
        self.tabs.setTabText(1, tr("tab_cake"))
        self.tabs.setTabText(2, tr("tab_1d"))
        self.tabs.setTabText(3, tr("tab_log"))
        # 各阶段面板静态控件
        self.gb0.setTitle(tr("s0_title"))
        self.drop_zone.setText(tr("s0_drop"))
        self.b_pick0.setText(tr("s0_pick"))
        self.demo_box.setTitle(tr("s0_demo"))
        self.b_demo_ceria.setText(tr("s0_demo_ceria"))
        self.b_demo_alumina.setText(tr("s0_demo_alumina"))
        self.b_next0.setText(tr("s0_next"))
        self.gb1.setTitle(tr("s1_title"))
        self.chk_gaps.setText(tr("s1_gaps"))
        self.chk_bad.setText(tr("s1_bad"))
        self.chk_bs.setText(tr("s1_bs"))
        self.b_extm.setText(tr("s1_extm"))
        self.b_clrm.setText(tr("s1_clear"))
        self.b_prev_mask.setText(tr("s1_preview"))
        self.sp_box.setTitle(tr("s1_spbox"))
        self.b_sp.setText(tr("s1_sp_pick"))
        self.b_sp_clr.setText(tr("s1_sp_off"))
        self.b_next1.setText(tr("s1_next"))
        self.gb2.setTitle(tr("s2_title"))
        self.lbl_px1.setText(tr("s2_px1"))
        self.lbl_px2.setText(tr("s2_px2"))
        self.lbl_lam.setText(tr("s2_lam"))
        self.lbl_phase.setText(tr("s2_phase"))
        self.lbl_dref.setText(tr("s2_dref"))
        self.e_px2.setPlaceholderText(tr("s2_px2_ph"))
        self.e_lam.setPlaceholderText(tr("s2_lam_ph"))
        self.e_dref.setPlaceholderText(tr("s2_dref_ph"))
        self.cb_phase.setItemText(0, tr("s2_phase_auto"))
        self.b_imp.setText(tr("s2_imp"))
        self.b_go.setText(tr("s2_go"))
        self.gb3.setTitle(tr("s3_title"))
        self.b_t2d.setText(tr("s3_b2d"))
        self.b_tcake.setText(tr("s3_bcake"))
        self.b_t1d.setText(tr("s3_b1d"))
        self.b_next3.setText(tr("s3_next"))
        self.gb4.setTitle(tr("s4_title"))
        self.b_exp_poni.setText(tr("s4_poni"))
        self.b_exp_fig.setText(tr("s4_fig"))
        self.b_copy_txt.setText(tr("s4_copy"))
        # 状态相关动态文案 + 流程图卡片 + 状态栏 + 图表标题
        self._refresh_ext_mask_lbl()
        self._refresh_spline_lbl()
        self._refresh_file_info()
        self._refresh_calib_mode()
        self._refresh_qa()
        self._refresh_export_label()
        self._refresh_cards()
        self._refresh_status()
        self._redraw_plots()

    # ---- 动态文案的统一再生成（供事件回调与 _retranslate 共用） ----
    def _mode_s(self):
        if not self.paths:
            return tr("st1_s")
        if self._n_files >= 2:
            return tr("mode_multi", n=self._n_files, name=self._first_name)
        return tr("mode_single", name=self._first_name)

    def _refresh_file_info(self):
        if not self.paths:
            self.lbl_file_info.setText(tr("s0_nofile"))
            return
        img = self.img_preview
        w, h = (img.shape[1], img.shape[0]) if img is not None else (0, 0)
        self.lbl_file_info.setText(
            tr("finfo", mode=self._mode_s(), w=w, h=h,
               src=self._meta_src or tr("meta_none")))

    def _refresh_calib_mode(self):
        if not self.paths:
            self.lbl_calib_mode.setText(tr("s2_mode_wait"))
        elif self._n_files >= 2:
            self.lbl_calib_mode.setText(tr("s2_mode_cur", mode=tr("mode_multi_lbl")))
        else:
            self.lbl_calib_mode.setText(tr("s2_mode_cur", mode=tr("mode_single_lbl")))

    def _refresh_ext_mask_lbl(self):
        if self.ext_mask is None:
            self.lbl_ext_mask.setText(tr("s1_extm_none"))
        else:
            n = int(np.asarray(self.ext_mask).sum())
            self.lbl_ext_mask.setText(
                tr("s1_extm_set", name=self._ext_mask_name or "?", n=n))

    def _refresh_spline_lbl(self):
        if self.spline_path:
            self.lbl_spline.setText(tr("s1_sp_set", name=os.path.basename(self.spline_path)))
        else:
            self.lbl_spline.setText(tr("s1_sp_none"))

    def _refresh_export_label(self):
        if self._exp_last:
            kind, p = self._exp_last
            self.lbl_export.setText(tr("s4_exp_poni" if kind == "poni" else "s4_exp_fig", p=p))
        else:
            self.lbl_export.setText(tr("s4_hint"))

    def _refresh_qa(self):
        if not self.payload:
            self.txt_qa.setPlainText(tr("s3_reset" if self._qa_state == "reset" else "s3_wait"))
            return
        self.txt_qa.setPlainText("\n".join(self._qa_lines()))

    def _refresh_status(self):
        if self.payload:
            self.status_lbl.setText(tr(
                "status_done", ph=self.payload["res"].get("phase"),
                d=f"{self.payload['geom'][0] * 1000:.3f}",
                tier=self._loc_conf()["tier"]))
        elif self.paths:
            self.status_lbl.setText(tr("status_loaded", mode=self._mode_s()))
        else:
            self.status_lbl.setText(tr("status_idle"))

    # ---- 流程图卡片状态统一入口（同时维护快照以便重翻译） ----
    def _set_card(self, idx, state, summary=None):
        if 0 <= idx < len(self.cards):
            self._card_states[idx] = state if state in STATE_STYLE else "idle"
            self.cards[idx].set_status(state, summary)

    def _last_mask_px(self):
        for m in ((self.payload or {}).get("mask"), self.preview_mask):
            if m is not None and bool(np.asarray(m).any()):
                return int(m.sum())
        return 0

    def _loc_conf(self):
        conf = (self.payload or {}).get("conf")
        if not conf:
            return dict(tier="-", badge="-", rationale="", recommendation="", raw_tier="-")
        res = self.payload.get("res") or {}
        return localize_conf(conf, has_d_ref=bool(res.get("d_ref_ok", False)))

    _RUN_KEYS = {1: "sum_mask_run", 2: "sum_solve_run", 3: "sum_cake_run"}

    def _card_summary(self, i):
        st = self._card_states[i]
        if st == "error":
            return tr("sum_error")
        if st == "running" and i in self._RUN_KEYS:
            return tr(self._RUN_KEYS[i])
        if i == 0:
            return self._mode_s() if st == "done" else tr("st1_s")
        if i == 1:
            if st == "done":
                sp = f" + {os.path.basename(self.spline_path)}" if self.spline_path else ""
                return tr("sum_mask_done", m=f"{self._last_mask_px()} px", sp=sp)
            return tr("st2_s")
        if i == 2:
            if st == "done" and self.payload:
                return tr("sum_geom_done", ph=self.payload["res"].get("phase"),
                          d=f"{self.payload['geom'][0] * 1000:.2f}")
            return tr("st3_s")
        if i == 3:
            if st == "done" and self.payload:
                return tr("sum_cake_done", tier=self._loc_conf()["tier"],
                          p=f"{self.payload['score']['ps_refined']:.2f}")
            return tr("st4_s")
        if i == 4:
            if st == "done" and self._exp_last:
                return tr("card_exported", name=os.path.basename(self._exp_last[1]))
            if self.payload:  # 流水线已跑完、尚未导出 → 保持“点击导出”提示
                return tr("sum_export_idle")
        return tr("st5_s")

    def _refresh_cards(self):
        for i, (title, sub) in enumerate(_stages()):
            self.cards[i].set_texts(title, self._card_summary(i))
            self.cards[i].set_status(self._card_states[i])
        self.select_stage(self.stack.currentIndex())  # 重新应用选中态样式

    # ---- QA 面板文本生成（完成回调与重翻译共用） ----
    def _qa_lines(self):
        payload = self.payload
        if not payload:
            return [tr("s3_wait")]
        res = payload["res"]
        sc = payload["score"]
        dist_m, poni1_m, poni2_m, rot1, rot2 = payload["geom"]
        px1_m, px2_m = payload["px1_m"], payload["px2_m"]
        lam_A = payload["lam_A"]
        ph = res.get("phase")
        cx_nat = poni2_m / px2_m - 0.5
        cy_nat = poni1_m / px1_m - 0.5
        loc = self._loc_conf()
        dq_s = f"{sc['dq_med']:.4f} Å⁻¹" if np.isfinite(sc["dq_med"]) else "N/A"
        m_v = int(res.get("m", res["sol"][3]))
        rms_v = float(res.get("rms", res["sol"][4]))
        return [
            f"{tr('qa_phase')}: {ph}",
            f"{tr('qa_dist')}: {dist_m * 1000:.3f} mm",
            f"{tr('qa_lam')}: {lam_A:.5f} Å",
            f"{tr('qa_center')}: ({cx_nat:.2f}, {cy_nat:.2f}) px",
            f"{tr('qa_poni')}: ({poni1_m * 1000:.3f}, {poni2_m * 1000:.3f}) mm",
            f"{tr('qa_rot')}: {rot1 * 1e3:+.2f} / {rot2 * 1e3:+.2f} mrad",
            f"{tr('qa_match')}: " + tr("qa_match_v", m=m_v, rms=f"{rms_v:.3f}"),
            f"{tr('qa_cake')}: " + tr("qa_cake_v", p=f"{sc['ps_refined']:.3f}",
                                       n=sc["n_peak_hit"], dq=dq_s),
            f"{tr('qa_conf')}: {loc['tier']} ({loc['badge']})",
            f"{tr('qa_rationale')}: {loc['rationale']}",
            f"{tr('qa_reco')}: {loc['recommendation']}",
        ]

    def _redraw_plots(self):
        """语言切换后按当前数据重画三张图（标题/图例随语言更新）。"""
        if self.payload:
            payload = self.payload
            res = payload["res"]
            ph = res.get("phase")
            qs = phase_qs(ph) if ph else np.array([], dtype=float)
            dist_m, poni1_m, poni2_m, _, _ = payload["geom"]
            px1_m, px2_m = payload["px1_m"], payload["px2_m"]
            cx_nat = poni2_m / px2_m - 0.5
            cy_nat = poni1_m / px1_m - 0.5
            D_px_mean = dist_m / (0.5 * (px1_m + px2_m))
            self._draw_2d(payload["img"], mask=payload["mask"], qs=qs,
                          cx_nat=cx_nat, cy_nat=cy_nat, D_px=D_px_mean,
                          lam_A=payload["lam_A"], phase=ph, dist_m=dist_m)
            self._draw_cake_tab(payload["cake"], qs, ph)
            self._draw_1d_tab(payload["cake"], qs, payload["score"])
        elif self.img_preview is not None:
            m = self.preview_mask if self.preview_mask is not None else self.ext_mask
            self._draw_2d(self.img_preview, mask=m)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.txt_log.appendPlainText(f"[{ts}] {msg}")

    def _refresh_phase_combo(self, select_name=None):
        cur = select_name or self.cb_phase.currentText()
        self.cb_phase.clear()
        self.cb_phase.addItem(tr("s2_phase_auto"))
        for ph in PHASE_NAMES:
            self.cb_phase.addItem(ph)
        if select_name:
            idx = self.cb_phase.findText(select_name)
            if idx >= 0:
                self.cb_phase.setCurrentIndex(idx)
        elif cur:
            idx = self.cb_phase.findText(cur)
            if idx >= 0:
                self.cb_phase.setCurrentIndex(idx)

    def _on_drag_enter(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def _on_drop(self, ev):
        urls = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.load_paths(urls)

    def pick_images(self):
        ps, _ = QFileDialog.getOpenFileNames(self, tr("dlg_pick_img"), "", tr("f_img"))
        if ps:
            self.load_paths(ps)

    def load_paths(self, paths):
        paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            return
        try:
            img0 = auto_offset(load_any(paths[0]))
        except Exception as exc:
            QMessageBox.warning(self, tr("msg_openfail"), str(exc))
            return

        self.paths = paths
        self.img_preview = img0
        self.preview_mask = None
        self.payload = None
        self.e_dref.clear()
        self.b_exp_poni.setEnabled(False)
        self.b_exp_fig.setEnabled(False)
        self.b_copy_txt.setEnabled(False)
        self.b_run_all.setEnabled(True)

        meta = sniff_meta(paths[0])
        if "px" in meta:
            self.e_px1.setText(f"{meta['px']:g}")
            self.e_px2.setText("")
        if "lam" in meta:
            self.cb_lam_unit.setCurrentIndex(0)
            self.e_lam.setText(f"{meta['lam']:g}")

        n = len(paths)
        bname = os.path.basename(paths[0])
        mode_s = tr("mode_single", name=bname) if n == 1 else tr("mode_multi", n=n, name=bname)
        src_s = meta.get("src") or tr("meta_none")
        # 记录快照供语言切换后 _retranslate 再生成
        self._n_files = n
        self._first_name = bname
        self._meta_src = meta.get("src")
        self._qa_state = "wait"
        self._refresh_file_info()
        self._refresh_calib_mode()

        self._set_card(0, "done", mode_s)
        for i in range(1, 5):
            self._card_states[i] = "idle"
        self._refresh_cards()
        self._draw_2d(img0, mask=self.ext_mask)
        self.tabs.setCurrentIndex(0)
        self._log(tr("log1_loaded", mode=mode_s, shape=img0.shape, meta=src_s))
        self.status_lbl.setText(tr("status_loaded", mode=mode_s))

    def load_demo_ceria(self):
        p = os.path.join(_data_root(), "real", "MX_ceria", "ceria_250_1_0001.cbf")
        if not os.path.isfile(p):
            QMessageBox.warning(self, tr("msg_nodemo"), p)
            return
        self.clear_spline()
        self.load_paths([p])
        self.cb_phase.setCurrentIndex(0)

    def load_demo_alumina(self):
        al_dir = os.path.join(_data_root(), "real", "alumina")
        ps = [os.path.join(al_dir, f"al2o3_{i:04d}.edf") for i in range(5)]
        ps = [p for p in ps if os.path.isfile(p)]
        if len(ps) < 2:
            QMessageBox.warning(self, tr("msg_nodemo"), al_dir)
            return
        self.load_paths(ps)
        poni_ref = os.path.join(al_dir, "al2o3_00_max_51_frames.poni")
        if os.path.isfile(poni_ref):
            kv = dict(l.split(":", 1) for l in open(poni_ref) if ":" in l and not l.startswith("#"))
            if "PixelSize1" in kv:
                self.e_px1.setText(f"{float(kv['PixelSize1']) * 1e6:.4f}")
            if "PixelSize2" in kv:
                self.e_px2.setText(f"{float(kv['PixelSize2']) * 1e6:.4f}")
            if "Wavelength" in kv:
                self.cb_lam_unit.setCurrentIndex(0)
                self.e_lam.setText(f"{float(kv['Wavelength']) * 1e10:.5f}")
        sp = os.path.join(al_dir, "distorsion_2x2.spline")
        if os.path.isfile(sp):
            self.spline_path = sp
            self.lbl_spline.setText(tr("s1_sp_set", name=os.path.basename(sp)))
        self.cb_phase.setCurrentIndex(0)

    def pick_ext_mask(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("dlg_pick_mask"), "", tr("f_mask"))
        if not p:
            return
        try:
            m = load_mask_file(p)
            if self.img_preview is not None and m.shape != self.img_preview.shape:
                QMessageBox.warning(
                    self, tr("msg_shape"),
                    tr("msg_shape_txt", m=m.shape, i=self.img_preview.shape))
                return
            self.ext_mask = m
            self._ext_mask_name = os.path.basename(p)
            self._refresh_ext_mask_lbl()
        except Exception as exc:
            QMessageBox.warning(self, tr("msg_loadfail"), str(exc))

    def clear_ext_mask(self):
        self.ext_mask = None
        self._ext_mask_name = None
        self._refresh_ext_mask_lbl()

    def preview_auto_mask(self):
        if self.img_preview is None:
            QMessageBox.information(self, tr("msg_hint"), tr("msg_noimg"))
            return
        m, st = build_auto_mask(
            self.img_preview,
            include_gaps=self.chk_gaps.isChecked(),
            include_bad=self.chk_bad.isChecked(),
            include_beamstop=self.chk_bs.isChecked(),
        )
        if self.ext_mask is not None and self.ext_mask.shape == m.shape:
            m = m | self.ext_mask
        self.preview_mask = m
        self._set_card(1, "done", tr("sum_mask_done", m=f"{int(m.sum())} px", sp=""))
        self._draw_2d(self.img_preview, mask=m)
        self.tabs.setCurrentIndex(0)
        self._log(tr("log2_preview", n=int(m.sum()), p=f"{st['frac']*100:.1f}"))

    def pick_spline(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("dlg_pick_spline"), "", tr("f_spl"))
        if p:
            self.spline_path = p
            self._refresh_spline_lbl()

    def clear_spline(self):
        self.spline_path = None
        self._refresh_spline_lbl()

    def import_custom_phase(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("dlg_imp_phase"), "", tr("f_phase"))
        if not p:
            return
        try:
            ph_name = load_and_register_phase(p)
            self._refresh_phase_combo(select_name=ph_name)
            qs = phase_qs(ph_name)
            self._log(tr("log3_import", ph=ph_name, n=len(qs)))
        except Exception as exc:
            QMessageBox.critical(self, tr("msg_impfail"), str(exc))

    def _read_lam_A(self):
        txt = self.e_lam.text().strip()
        if not txt:
            return None
        val = float(txt)
        if val <= 0:
            raise ValueError(tr("err_lam_pos"))
        return self.HC_KEV_A / val if self.cb_lam_unit.currentIndex() == 1 else val

    def run_all(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.paths:
            QMessageBox.information(self, tr("msg_hint"), tr("msg_nofile"))
            self.select_stage(0)
            return
        try:
            px1_um = float(self.e_px1.text().strip())
            px2_txt = self.e_px2.text().strip()
            px2_um = float(px2_txt) if px2_txt else px1_um
            lam_A = self._read_lam_A()
            if lam_A is None:
                raise ValueError(tr("err_need_lam"))
            dref_txt = self.e_dref.text().strip()
            d_ref_mm = float(dref_txt) if dref_txt else None
            if d_ref_mm is not None and d_ref_mm <= 0:
                raise ValueError(tr("err_dref_pos"))
        except Exception as exc:
            QMessageBox.warning(self, tr("msg_badparam"), str(exc))
            self.select_stage(2)
            return

        phase_opt = None if self.cb_phase.currentIndex() == 0 else [self.cb_phase.currentText()]
        self.pb.setVisible(True)
        self.b_run_all.setEnabled(False)
        self.status_lbl.setText(tr("status_running"))

        self.worker = WorkflowWorker(
            self.paths, px1_um, px2_um, lam_A, phase_opt=phase_opt,
            auto_gaps=self.chk_gaps.isChecked(),
            auto_bad=self.chk_bad.isChecked(),
            auto_bs=self.chk_bs.isChecked(),
            ext_mask=self.ext_mask,
            spline_path=self.spline_path,
            d_ref_mm=d_ref_mm,
        )
        self.worker.stage_changed.connect(self._on_stage_changed)
        self.worker.log_line.connect(self._log)
        self.worker.finished_all.connect(self._on_finished_all)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_stage_changed(self, idx, state, summary):
        if 0 <= idx < len(self.cards):
            self._set_card(idx, state, summary)
        if state == "running":
            self.status_lbl.setText(
                tr("status_stage_run", stage=tr("st%d_t" % (idx + 1)), summary=summary))

    def _on_failed(self, idx, tb):
        self.pb.setVisible(False)
        self.b_run_all.setEnabled(True)
        self._set_card(idx, "error", tr("sum_error"))
        stage_t = tr("st%d_t" % (idx + 1))
        self._log(tr("log_err", stage=stage_t, tb=tb))
        self.status_lbl.setText(tr("status_fail", stage=stage_t))
        self.tabs.setCurrentIndex(3)

    def _on_finished_all(self, payload):
        self.pb.setVisible(False)
        self.b_run_all.setEnabled(True)
        self.payload = payload

        img = payload["img"]
        mask = payload["mask"]
        res = payload["res"]
        cake = payload["cake"]
        sc = payload["score"]
        conf = payload["conf"]
        dist_m, poni1_m, poni2_m, rot1, rot2 = payload["geom"]
        px1_m, px2_m = payload["px1_m"], payload["px2_m"]
        lam_A = payload["lam_A"]
        ph = res.get("phase")
        qs = phase_qs(ph) if ph else np.array([], dtype=float)

        cx_nat = poni2_m / px2_m - 0.5
        cy_nat = poni1_m / px1_m - 0.5
        D_px_mean = dist_m / (0.5 * (px1_m + px2_m))

        self._draw_2d(img, mask=mask, qs=qs, cx_nat=cx_nat, cy_nat=cy_nat,
                      D_px=D_px_mean, lam_A=lam_A, phase=ph, dist_m=dist_m)
        self._draw_cake_tab(cake, qs, ph)
        self._draw_1d_tab(cake, qs, sc)

        dq_s = f"{sc['dq_med']:.4f} Å⁻¹" if np.isfinite(sc["dq_med"]) else "N/A"
        self.txt_qa.setPlainText("\n".join(self._qa_lines()))
        self._qa_state = "full"

        self.b_exp_poni.setEnabled(True)
        self.b_exp_fig.setEnabled(True)
        self.b_copy_txt.setEnabled(True)
        self.select_stage(3)
        self.tabs.setCurrentIndex(1)
        self.status_lbl.setText(tr(
            "status_done", ph=ph, d=f"{dist_m*1000:.3f}", tier=self._loc_conf()["tier"]))

    def _draw_2d(self, img, mask=None, qs=None, cx_nat=None, cy_nat=None,
                 D_px=None, lam_A=None, phase=None, dist_m=None):
        canvas, s, _, _ = to_canvas(img)
        self.ax_2d.clear()
        disp = np.log1p(np.clip(canvas, 0.0, None))
        fin = disp[np.isfinite(disp)]
        vlo, vhi = np.percentile(fin, [5.0, 99.2]) if fin.size else (0.0, 1.0)
        self.ax_2d.imshow(disp, origin="upper", cmap="gray", vmin=float(vlo), vmax=float(vhi))
        if mask is not None and np.asarray(mask).shape == img.shape:
            self.ax_2d.imshow(_mask_overlay(mask, canvas, s, alpha=0.35), origin="upper")
        if cx_nat is not None and cy_nat is not None and D_px is not None and lam_A is not None:
            cx_c = (float(cx_nat) + 0.5) / s - 0.5
            cy_c = (float(cy_nat) + 0.5) / s - 0.5
            D_c = float(D_px) / s
            self.ax_2d.plot([cx_c], [cy_c], "r+", ms=12, mew=2.0)
            for r_c in _r_model(np.asarray(qs, float), lam_A, D_c):
                if 5.0 < r_c < 1.45 * max(canvas.shape):
                    self.ax_2d.add_patch(Circle((cx_c, cy_c), float(r_c), fill=False,
                                                ec="#009E73", lw=0.9, alpha=0.8))
            self.ax_2d.set_title(tr("t_2d_rings", ph=phase, d=f"{dist_m*1000:.2f}"), fontsize=10)
        else:
            self.ax_2d.set_title(tr("t_2d_raw"), fontsize=10)
        self.fig_2d.tight_layout()
        self.cv_2d.draw_idle()

    def _draw_cake_tab(self, cake, qs, phase):
        draw_cake(self.ax_cake, cake, qs=qs, title=tr("t_cake", ph=phase))
        self.fig_cake.tight_layout()
        self.cv_cake.draw_idle()

    def _draw_1d_tab(self, cake, qs, sc):
        self.ax_1d.clear()
        q_1d = cake["q_axis"]
        p_1d = np.nanmean(cake["cake"], axis=0)
        fin = np.isfinite(p_1d)
        if fin.any():
            self.ax_1d.plot(q_1d[fin], np.log1p(np.clip(p_1d[fin], 0.0, None)), color="#0072B2", lw=1.2)
        for q0 in qs:
            if q_1d[0] <= q0 <= q_1d[-1]:
                self.ax_1d.axvline(float(q0), color="#D55E00", lw=0.8, ls="--", alpha=0.75)
        self.ax_1d.set_xlabel(r"q (Å$^{-1}$)")
        self.ax_1d.set_ylabel("log1p(I)")
        dq_s = f"{sc['dq_med']:.4f} Å" + r"$^{-1}$" if np.isfinite(sc["dq_med"]) else "N/A"
        self.ax_1d.set_title(tr("t_1d", p=f"{sc['ps_refined']:.3f}", dq=dq_s), fontsize=10)
        self.fig_1d.tight_layout()
        self.cv_1d.draw_idle()

    def export_poni(self):
        if not self.payload:
            return
        base = os.path.splitext(os.path.basename(self.paths[0]))[0]
        p, _ = QFileDialog.getSaveFileName(self, tr("dlg_save_poni"), f"{base}_calibrated.poni", tr("f_poni"))
        if not p:
            return
        try:
            d_m, p1_m, p2_m, r1, r2 = self.payload["geom"]
            write_poni(
                p, d_m, p1_m, p2_m, r1, r2,
                self.payload["lam_A"], self.payload["px1_m"], self.payload["px2_m"],
                spline=self.payload["spline"], mask=self.payload["mask"],
                note=f"Exported from gui_workflow.py ({self.payload['res'].get('phase')})",
            )
            self._exp_last = ("poni", p)
            self._set_card(4, "done", tr("card_exported", name=os.path.basename(p)))
            self.lbl_export.setText(tr("s4_exp_poni", p=p))
            self._log(tr("log5_poni", p=p))
        except Exception as exc:
            QMessageBox.critical(self, tr("msg_expfail"), str(exc))

    def export_diag_fig(self):
        if not self.payload:
            return
        base = os.path.splitext(os.path.basename(self.paths[0]))[0]
        p, _ = QFileDialog.getSaveFileName(self, tr("dlg_save_fig"), f"{base}_{tr('fig_suffix')}.png", tr("f_png"))
        if not p:
            return
        try:
            export_diagnostic_figure(
                p, self.payload["img"], self.payload["res"],
                self.payload["px1_m"], px2_m=self.payload["px2_m"],
                mask=self.payload["mask"], spline=self.payload["spline"],
            )
            self._exp_last = ("fig", p)
            self._set_card(4, "done", tr("card_exported", name=os.path.basename(p)))
            self.lbl_export.setText(tr("s4_exp_fig", p=p))
            self._log(tr("log5_fig", p=p))
        except Exception as exc:
            QMessageBox.critical(self, tr("msg_expfail"), str(exc))

    def copy_summary(self):
        if not self.payload:
            return
        QApplication.clipboard().setText(self.txt_qa.toPlainText())
        self.status_lbl.setText(tr("status_copied"))

    def reset_workflow(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.paths = []
        self.img_preview = None
        self.ext_mask = None
        self.preview_mask = None
        self.payload = None
        self.clear_spline()
        self.clear_ext_mask()
        # 快照一并复位，保证后续语言切换时动态文案回到初始态
        self._card_states = ["idle"] * 5
        self._n_files = 0
        self._first_name = ""
        self._meta_src = None
        self._qa_state = "reset"
        self._exp_last = None
        self._refresh_file_info()
        self._refresh_calib_mode()
        self.txt_qa.setPlainText(tr("s3_reset"))
        self.txt_log.clear()
        self.b_run_all.setEnabled(False)
        self.b_exp_poni.setEnabled(False)
        self.b_exp_fig.setEnabled(False)
        self.b_copy_txt.setEnabled(False)
        self._refresh_cards()
        self._refresh_export_label()
        for ax, cv in ((self.ax_2d, self.cv_2d), (self.ax_cake, self.cv_cake), (self.ax_1d, self.cv_1d)):
            ax.clear()
            cv.draw_idle()
        self.select_stage(0)
        self.status_lbl.setText(tr("status_reset"))


def _selftest():
    """打包自检：无头验证 frozen exe 内全部关键依赖与流程图主窗口构建。"""
    log = os.path.join(os.environ.get("TEMP", "."), "PyFAI-Workflow-selftest.log")
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import torchfai.phases
        import fabio
        import pyFAI.geometry
        import pyFAI.detectors
        import matplotlib
        from PyQt5.QtWidgets import QApplication
        import auto_mask
        import cake_report
        import confidence_map
        import custom_phase
        import multiframe_calib
        import solver_core

        assert len(torchfai.phases.phase_qs("Al2O3")) >= 5
        app = QApplication.instance() or QApplication([])
        theme.apply_app(app, "light")
        win = WorkflowWindow()
        assert len(win.cards) == 5
        # i18n 冒烟：切到英语全量重翻译并抽查关键控件，随后还原语言
        prev_lang = get_lang()
        set_lang("en", persist=False)
        win._retranslate()
        assert win.windowTitle() == tr("app_title")
        assert win.gb0.title() == tr("s0_title")
        assert win.b_run_all.text() == tr("run_all")
        assert win.tabs.tabText(3) == tr("tab_log")
        win.reset_workflow()
        assert win.status_lbl.text() == tr("status_reset")
        set_lang(prev_lang, persist=False)
        win._retranslate()
        msg = (
            f"SELFTEST_OK phases={len(torchfai.phases.PHASE_NAMES)} "
            f"cards={len(win.cards)} lang={get_lang()} "
            f"pyFAI={getattr(pyFAI, 'version', 'ok')}"
        )
        with open(log, "w", encoding="utf-8") as f:
            f.write(msg + "\n")
        if sys.stdout is not None:
            print(msg, flush=True)
    except Exception:
        with open(log, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        if sys.stderr is not None:
            traceback.print_exc()
        sys.exit(1)


def main():
    if "--selftest" in sys.argv:
        _selftest()
        return
    app = QApplication(sys.argv)
    theme.apply_app(app, "light")
    win = WorkflowWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
