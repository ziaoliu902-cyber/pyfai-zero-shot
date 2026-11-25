# -*- coding: utf-8 -*-
"""gui_workflow.py：流程图式交互标定工作台。

将衍射几何盲解封装为 5 个可视化流程图节点：
  ① 数据导入 → ② 掩膜与畸变 → ③ 自动标定 → ④ Cake 体检 → ⑤ 导出结果
点击任意节点可查看/配置该步参数与输出摘要，也可点击【一键运行全流程】自动贯通执行。
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
from auto_mask import build_auto_mask
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

IMG_FILTER = "衍射图 (*.edf *.cbf *.h5 *.hdf5 *.tif *.tiff *.mar* *.img *.png *.jpg);;所有 (*.*)"
SPL_FILTER = "pyFAI 畸变文件 (*.spline);;所有 (*.*)"
MASK_FILTER = "Mask 文件 (*.npy *.h5 *.hdf5 *.edf *.tif *.tiff *.cbf);;所有 (*.*)"
PHASE_FILTER = "标样文件 (*.D *.d *.cif);;所有 (*.*)"

STAGES = [
    ("1. 数据导入", "选择单帧或多帧衍射图"),
    ("2. 掩膜与畸变", "自动屏蔽坏点/束挡与可选畸变"),
    ("3. 自动标定", "物相识别与几何参数求解"),
    ("4. Cake 体检", "展开均匀度与置信度核验"),
    ("5. 导出结果", "保存 .poni / 2×2 体检图"),
]

STATE_STYLE = {
    "idle": ("待运行", "#9CA3AF", "#F3F4F6"),
    "running": ("运行中", "#2563EB", "#DBEAFE"),
    "done": ("已完成", "#0A7D0A", "#DCFCE7"),
    "skipped": ("已跳过", "#6B7280", "#E5E7EB"),
    "error": ("失败", "#C2410C", "#FFEDD5"),
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
        self.lbl_badge = QLabel("待运行")
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

    def set_status(self, state, summary=None):
        self.state = state if state in STATE_STYLE else "idle"
        if summary is not None:
            self.lbl_sub.setText(str(summary))
        self._apply_style()

    def _apply_style(self):
        t = theme.TOKENS["light"]
        label, fg, bg = STATE_STYLE[self.state]
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
            self.stage_changed.emit(1, "running", "生成/合并掩膜…")
            self.log_line.emit(f"[步骤2] 准备掩膜与畸变校正 (文件数={len(self.paths)})")
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
                self.log_line.emit(
                    f"[步骤2] 自动掩膜: 缝={mst['n_gap']}, 坏点/热像素={mst['n_bad']}, "
                    f"束挡={mst['n_beamstop']}, 合计={mst['n_total']} px ({mst['frac']*100:.1f}%)"
                )
            else:
                self.log_line.emit("[步骤2] 自动掩膜已关闭")
            sp_tag = f" + {os.path.basename(self.spline_path)}" if self.spline_path else ""
            mask_tag = f"{int(mask.sum())} px" if mask is not None and mask.any() else "无"
            self.stage_changed.emit(1, "done", f"mask={mask_tag}{sp_tag}")

            # ---- Stage 3: 自动标定 ----
            cur_stage = 2
            self.stage_changed.emit(2, "running", "求解几何与物相…")
            if is_multi:
                self.log_line.emit(f"[步骤3] 启动多帧抗差联合标定 ({len(self.paths)} 帧)…")
                mf = calibrate_multiframe(
                    self.paths, px1_m, lam=self.lam_A, px2_m=px2_m,
                    phases=self.phase_opt, spline=self.spline_path, n_workers=1,
                )
                if not mf.get("ok"):
                    raise RuntimeError(mf.get("error", "多帧联合标定失败"))
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
                self.log_line.emit(
                    f"[步骤3] 多帧完成: 保留 {mf['n_kept']}/{mf['n_total']} 帧 (暗场 {mf['n_dark']}, 离群 {mf['n_outliers']}) | "
                    f"相={ph} | D={mf['dist_mm']:.3f} mm | rot=({rot1*1e3:+.2f}, {rot2*1e3:+.2f}) mrad"
                )
            else:
                img_work = first_raw
                self.log_line.emit("[步骤3] 启动单帧几何求解…")
                res = solve_image(
                    img_work, px1_m, lam=self.lam_A, phases=self.phase_opt,
                    px2_m=px2_m, mask=mask, d_ref_mm=self.d_ref_mm,
                )
                if not res:
                    raise RuntimeError("求解失败：未能匹配足够衍射环，请检查像素尺寸、波长或改用多帧堆叠。")
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
                self.log_line.emit(
                    f"[步骤3] 单帧完成: 相={ph} | D={dist_m*1000:.3f} mm | "
                    f"m={res['m']} | rms={res['rms']:.2f} px | rot=({rot1*1e3:+.2f}, {rot2*1e3:+.2f}) mrad"
                )
            self.stage_changed.emit(2, "done", f"{ph} | D={dist_m*1000:.2f} mm")

            # ---- Stage 4: Cake 体检 ----
            cur_stage = 3
            self.stage_changed.emit(3, "running", "展开 Cake 与质量评分…")
            self.log_line.emit("[步骤4] 正在生成 q-χ Cake 展开图与 SNIP 峰位核验…")
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
            self.log_line.emit(
                f"[步骤4] 体检完成: refined={sc['ps_refined']:.3f} | |Δq|_med={dq_txt} ({sc['n_peak_hit']} 峰) | "
                f"置信度={conf['tier']} ({conf['badge']})"
            )
            self.stage_changed.emit(3, "done", f"{conf['tier']} | refined={sc['ps_refined']:.2f}")
            self.stage_changed.emit(4, "idle", "点击导出 .poni 或诊断图")

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
        self.setWindowTitle("pyFAI-Blind · 流程图式标定工作台")
        self.resize(1340, 820)
        self.paths = []
        self.img_preview = None
        self.ext_mask = None
        self.preview_mask = None
        self.spline_path = None
        self.payload = None
        self.worker = None

        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setContentsMargins(10, 10, 10, 8)
        root_lay.setSpacing(8)

        # ---- 顶部交互式流程图工具条 ----
        top_box = QGroupBox("标定工作流（点击节点可配置参数或查看详情）")
        top_lay = QHBoxLayout(top_box)
        top_lay.setContentsMargins(10, 12, 10, 10)
        top_lay.setSpacing(6)

        self.cards = []
        for idx, (title, sub) in enumerate(STAGES):
            if idx > 0:
                top_lay.addWidget(FlowArrow())
            card = FlowNodeCard(idx, title, sub)
            card.clicked.connect(self.select_stage)
            self.cards.append(card)
            top_lay.addWidget(card, 1)

        ctrl_col = QVBoxLayout()
        ctrl_col.setSpacing(6)
        self.b_run_all = QPushButton("▶ 一键运行全流程")
        self.b_run_all.setProperty("primary", True)
        self.b_run_all.setMinimumHeight(34)
        self.b_run_all.setEnabled(False)
        self.b_run_all.clicked.connect(self.run_all)
        self.b_reset = QPushButton("重置")
        self.b_reset.clicked.connect(self.reset_workflow)
        ctrl_col.addWidget(self.b_run_all)
        ctrl_col.addWidget(self.b_reset)
        top_lay.addLayout(ctrl_col)
        root_lay.addWidget(top_box)

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

        self.tabs.addTab(tab_2d, "2D 衍射图与环叠图")
        self.tabs.addTab(tab_cake, "q-χ Cake 展开图")
        self.tabs.addTab(tab_1d, "1D 径向积分谱")
        self.tabs.addTab(self.txt_log, "运行日志")

        splitter.addWidget(left_wrap)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_lay.addWidget(splitter, 1)

        self.status_lbl = QLabel("请先在步骤 1 导入衍射图，或直接点击示例数据试跑。")
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
        box = QGroupBox("步骤 1 · 数据导入（单帧或同工位多帧）")
        lay = QVBoxLayout(box)

        self.drop_zone = QLabel("将衍射图拖放到此处\n（选 1 张 = 单帧盲解 | 选 ≥2 张 = 多帧抗差联合）")
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

        b_pick = QPushButton("选择衍射图文件 (单帧 / 多选)…")
        b_pick.setProperty("primary", True)
        b_pick.clicked.connect(self.pick_images)
        lay.addWidget(b_pick)

        self.lbl_file_info = QLabel("尚未导入文件")
        self.lbl_file_info.setWordWrap(True)
        self.lbl_file_info.setProperty("muted", True)
        lay.addWidget(self.lbl_file_info)

        demo_box = QGroupBox("快速体验内置真图")
        dl = QVBoxLayout(demo_box)
        self.b_demo_ceria = QPushButton("载入 Pilatus6M 示例 (ceria_250)")
        self.b_demo_ceria.clicked.connect(self.load_demo_ceria)
        self.b_demo_alumina = QPushButton("载入 FReLoN 5帧示例 (alumina + spline)")
        self.b_demo_alumina.clicked.connect(self.load_demo_alumina)
        dl.addWidget(self.b_demo_ceria)
        dl.addWidget(self.b_demo_alumina)
        lay.addWidget(demo_box)

        lay.addStretch(1)
        b_next = QPushButton("下一步：掩膜与畸变 →")
        b_next.clicked.connect(lambda: self.select_stage(1))
        lay.addWidget(b_next)
        return box

    def _build_stage1(self):
        box = QGroupBox("步骤 2 · 掩膜与光学畸变校正")
        lay = QVBoxLayout(box)

        self.chk_gaps = QCheckBox("自动屏蔽模块缝 / 死区间隙")
        self.chk_gaps.setChecked(True)
        self.chk_bad = QCheckBox("自动屏蔽坏点 / 负值 / 热像素")
        self.chk_bad.setChecked(True)
        self.chk_bs = QCheckBox("自动识别束流挡盘与挡杆阴影")
        self.chk_bs.setChecked(True)
        lay.addWidget(self.chk_gaps)
        lay.addWidget(self.chk_bad)
        lay.addWidget(self.chk_bs)

        row_m = QHBoxLayout()
        b_extm = QPushButton("额外载入外部 Mask…")
        b_extm.clicked.connect(self.pick_ext_mask)
        b_clrm = QPushButton("清除")
        b_clrm.clicked.connect(self.clear_ext_mask)
        row_m.addWidget(b_extm, 1)
        row_m.addWidget(b_clrm)
        lay.addLayout(row_m)
        self.lbl_ext_mask = QLabel("外部 Mask：无")
        self.lbl_ext_mask.setProperty("muted", True)
        lay.addWidget(self.lbl_ext_mask)

        b_prev_mask = QPushButton("生成并预览自动 Mask")
        b_prev_mask.clicked.connect(self.preview_auto_mask)
        lay.addWidget(b_prev_mask)

        sp_box = QGroupBox("探测器光学畸变文件（可选 .spline）")
        sl = QVBoxLayout(sp_box)
        row_sp = QHBoxLayout()
        b_sp = QPushButton("选择 .spline 文件…")
        b_sp.clicked.connect(self.pick_spline)
        b_sp_clr = QPushButton("不用畸变校正")
        b_sp_clr.clicked.connect(self.clear_spline)
        row_sp.addWidget(b_sp, 1)
        row_sp.addWidget(b_sp_clr)
        sl.addLayout(row_sp)
        self.lbl_spline = QLabel("未载入（Pilatus / Eiger 平板探测器留空即可）")
        self.lbl_spline.setWordWrap(True)
        self.lbl_spline.setProperty("muted", True)
        sl.addWidget(self.lbl_spline)
        lay.addWidget(sp_box)

        lay.addStretch(1)
        b_next = QPushButton("下一步：自动标定 →")
        b_next.clicked.connect(lambda: self.select_stage(2))
        lay.addWidget(b_next)
        return box

    def _build_stage2(self):
        box = QGroupBox("步骤 3 · 标定参数与运行求解")
        lay = QVBoxLayout(box)

        grid = QGridLayout()
        grid.addWidget(QLabel("像素尺寸 Y (µm):"), 0, 0)
        self.e_px1 = QLineEdit("172")
        grid.addWidget(self.e_px1, 0, 1)

        grid.addWidget(QLabel("像素尺寸 X (µm):"), 1, 0)
        self.e_px2 = QLineEdit("")
        self.e_px2.setPlaceholderText("留空默认等于 Y")
        grid.addWidget(self.e_px2, 1, 1)

        grid.addWidget(QLabel("波长 / 能量:"), 2, 0)
        row_lam = QHBoxLayout()
        self.e_lam = QLineEdit("")
        self.e_lam.setPlaceholderText("例如 0.9724")
        self.cb_lam_unit = QComboBox()
        self.cb_lam_unit.addItems(["Å", "keV"])
        row_lam.addWidget(self.e_lam, 1)
        row_lam.addWidget(self.cb_lam_unit)
        grid.addLayout(row_lam, 2, 1)

        grid.addWidget(QLabel("标样物相:"), 3, 0)
        self.cb_phase = QComboBox()
        self._refresh_phase_combo()
        grid.addWidget(self.cb_phase, 3, 1)

        grid.addWidget(QLabel("近似距离 (mm):"), 4, 0)
        self.e_dref = QLineEdit("")
        self.e_dref.setPlaceholderText("可选，如 2600（少环/AgBh 消歧用）")
        grid.addWidget(self.e_dref, 4, 1)
        lay.addLayout(grid)

        b_imp = QPushButton("导入自定义标样 (.D / .cif)…")
        b_imp.clicked.connect(self.import_custom_phase)
        lay.addWidget(b_imp)

        self.lbl_calib_mode = QLabel("求解模式：待导入图像后自动判断（单帧 / 多帧联合）")
        self.lbl_calib_mode.setWordWrap(True)
        self.lbl_calib_mode.setProperty("muted", True)
        lay.addWidget(self.lbl_calib_mode)

        lay.addStretch(1)
        b_go = QPushButton("▶ 立即运行流水线")
        b_go.setProperty("primary", True)
        b_go.clicked.connect(self.run_all)
        lay.addWidget(b_go)
        return box

    def _build_stage3(self):
        box = QGroupBox("步骤 4 · Cake 展开与质量体检")
        lay = QVBoxLayout(box)

        self.txt_qa = QPlainTextEdit()
        self.txt_qa.setReadOnly(True)
        self.txt_qa.setPlainText("尚未运行标定。完成步骤 1~3 后将在此展示几何参数与体检指标。")
        lay.addWidget(self.txt_qa, 1)

        row_t = QHBoxLayout()
        b_t2d = QPushButton("看 2D 叠环")
        b_t2d.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        b_tcake = QPushButton("看 Cake")
        b_tcake.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        b_t1d = QPushButton("看 1D 谱")
        b_t1d.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        row_t.addWidget(b_t2d)
        row_t.addWidget(b_tcake)
        row_t.addWidget(b_t1d)
        lay.addLayout(row_t)

        b_next = QPushButton("下一步：导出结果 →")
        b_next.clicked.connect(lambda: self.select_stage(4))
        lay.addWidget(b_next)
        return box

    def _build_stage4(self):
        box = QGroupBox("步骤 5 · 导出标准成果")
        lay = QVBoxLayout(box)

        self.b_exp_poni = QPushButton("导出 .poni 标定文件（含 Mask / Spline）")
        self.b_exp_poni.setProperty("primary", True)
        self.b_exp_poni.setEnabled(False)
        self.b_exp_poni.clicked.connect(self.export_poni)

        self.b_exp_fig = QPushButton("导出 2×2 标定体检图 (.png / .pdf)")
        self.b_exp_fig.setEnabled(False)
        self.b_exp_fig.clicked.connect(self.export_diag_fig)

        self.b_copy_txt = QPushButton("复制参数摘要到剪贴板")
        self.b_copy_txt.setEnabled(False)
        self.b_copy_txt.clicked.connect(self.copy_summary)

        lay.addWidget(self.b_exp_poni)
        lay.addWidget(self.b_exp_fig)
        lay.addWidget(self.b_copy_txt)

        self.lbl_export = QLabel("运行全流程完成后即可在此一键导出。")
        self.lbl_export.setWordWrap(True)
        self.lbl_export.setProperty("muted", True)
        lay.addWidget(self.lbl_export)
        lay.addStretch(1)
        return box

    # ---- 交互逻辑 ----
    def select_stage(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, c in enumerate(self.cards):
            c.set_selected(i == idx)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.txt_log.appendPlainText(f"[{ts}] {msg}")

    def _refresh_phase_combo(self, select_name=None):
        cur = select_name or self.cb_phase.currentText()
        self.cb_phase.clear()
        self.cb_phase.addItem("全盲自动识别（不给标样相）")
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
        ps, _ = QFileDialog.getOpenFileNames(self, "选择衍射图（可多选）", "", IMG_FILTER)
        if ps:
            self.load_paths(ps)

    def load_paths(self, paths):
        paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            return
        try:
            img0 = auto_offset(load_any(paths[0]))
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
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
        mode_s = f"单帧 ({bname})" if n == 1 else f"多帧 ({n} 帧，首帧 {bname})"
        src_s = meta.get("src") or "无头信息，请在步骤 3 确认像素/波长"
        self.lbl_file_info.setText(f"已载入：{mode_s}\n尺寸：{img0.shape[1]} × {img0.shape[0]} px\n元数据：{src_s}")
        self.lbl_calib_mode.setText(
            f"当前将自动走：{'多帧抗差去暗场联合标定' if n >= 2 else '单帧几何盲解与倾角精化'}"
        )

        self.cards[0].set_status("done", mode_s)
        for i in range(1, 5):
            self.cards[i].set_status("idle", STAGES[i][1])
        self._draw_2d(img0, mask=self.ext_mask)
        self.tabs.setCurrentIndex(0)
        self._log(f"[步骤1] 已导入 {mode_s}, shape={img0.shape}, meta={src_s}")
        self.status_lbl.setText(f"已导入 {mode_s}，可直接点击顶部【▶ 一键运行全流程】。")

    def load_demo_ceria(self):
        p = os.path.join(_data_root(), "real", "MX_ceria", "ceria_250_1_0001.cbf")
        if not os.path.isfile(p):
            QMessageBox.warning(self, "找不到示例", p)
            return
        self.clear_spline()
        self.load_paths([p])
        self.cb_phase.setCurrentIndex(0)

    def load_demo_alumina(self):
        al_dir = os.path.join(_data_root(), "real", "alumina")
        ps = [os.path.join(al_dir, f"al2o3_{i:04d}.edf") for i in range(5)]
        ps = [p for p in ps if os.path.isfile(p)]
        if len(ps) < 2:
            QMessageBox.warning(self, "找不到示例", al_dir)
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
            self.lbl_spline.setText(f"已载入：{os.path.basename(sp)}")
        self.cb_phase.setCurrentIndex(0)

    def pick_ext_mask(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择外部 Mask", "", MASK_FILTER)
        if not p:
            return
        try:
            m = load_mask_file(p)
            if self.img_preview is not None and m.shape != self.img_preview.shape:
                QMessageBox.warning(self, "尺寸不匹配", f"Mask {m.shape} 与图像 {self.img_preview.shape} 不一致")
                return
            self.ext_mask = m
            self.lbl_ext_mask.setText(f"外部 Mask：{os.path.basename(p)} ({int(m.sum())} px)")
        except Exception as exc:
            QMessageBox.warning(self, "载入失败", str(exc))

    def clear_ext_mask(self):
        self.ext_mask = None
        self.lbl_ext_mask.setText("外部 Mask：无")

    def preview_auto_mask(self):
        if self.img_preview is None:
            QMessageBox.information(self, "提示", "请先在步骤 1 导入图像。")
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
        self.cards[1].set_status("done", f"mask={int(m.sum())} px")
        self._draw_2d(self.img_preview, mask=m)
        self.tabs.setCurrentIndex(0)
        self._log(f"[步骤2] 预览掩膜完成: 合计 {int(m.sum())} px ({st['frac']*100:.1f}%)")

    def pick_spline(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 .spline 畸变文件", "", SPL_FILTER)
        if p:
            self.spline_path = p
            self.lbl_spline.setText(f"已载入：{os.path.basename(p)}")

    def clear_spline(self):
        self.spline_path = None
        self.lbl_spline.setText("未载入（Pilatus / Eiger 平板探测器留空即可）")

    def import_custom_phase(self):
        p, _ = QFileDialog.getOpenFileName(self, "导入自定义标样 (.D / .cif)", "", PHASE_FILTER)
        if not p:
            return
        try:
            ph_name = load_and_register_phase(p)
            self._refresh_phase_combo(select_name=ph_name)
            qs = phase_qs(ph_name)
            self._log(f"[步骤3] 已导入自定义标样 {ph_name} ({len(qs)} 条衍射峰)")
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    def _read_lam_A(self):
        txt = self.e_lam.text().strip()
        if not txt:
            return None
        val = float(txt)
        if val <= 0:
            raise ValueError("波长/能量必须 > 0")
        return self.HC_KEV_A / val if self.cb_lam_unit.currentIndex() == 1 else val

    def run_all(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.paths:
            QMessageBox.information(self, "提示", "请先在步骤 1 导入衍射图。")
            self.select_stage(0)
            return
        try:
            px1_um = float(self.e_px1.text().strip())
            px2_txt = self.e_px2.text().strip()
            px2_um = float(px2_txt) if px2_txt else px1_um
            lam_A = self._read_lam_A()
            if lam_A is None:
                raise ValueError("请填写波长 (Å) 或光子能量 (keV)")
            dref_txt = self.e_dref.text().strip()
            d_ref_mm = float(dref_txt) if dref_txt else None
            if d_ref_mm is not None and d_ref_mm <= 0:
                raise ValueError("近似距离必须 > 0")
        except Exception as exc:
            QMessageBox.warning(self, "参数有误", str(exc))
            self.select_stage(2)
            return

        phase_opt = None if self.cb_phase.currentIndex() == 0 else [self.cb_phase.currentText()]
        self.pb.setVisible(True)
        self.b_run_all.setEnabled(False)
        self.status_lbl.setText("全流程运行中…")

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
            self.cards[idx].set_status(state, summary)
        if state == "running":
            self.status_lbl.setText(f"{STAGES[idx][0]}：{summary}")

    def _on_failed(self, idx, tb):
        self.pb.setVisible(False)
        self.b_run_all.setEnabled(True)
        self.cards[idx].set_status("error", "运行失败")
        self._log(f"[错误] {STAGES[idx][0]} 异常:\n{tb}")
        self.status_lbl.setText(f"{STAGES[idx][0]} 运行失败，请查看日志。")
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
        qa_lines = [
            f"标样物相        : {ph}",
            f"样品到探测器距离: {dist_m * 1000:.3f} mm",
            f"X 射线波长      : {lam_A:.5f} Å",
            f"原生束心 (x, y) : ({cx_nat:.2f}, {cy_nat:.2f}) px",
            f"PONI (p1, p2)   : ({poni1_m * 1000:.3f}, {poni2_m * 1000:.3f}) mm",
            f"倾角 rot1/rot2  : {rot1 * 1e3:+.2f} / {rot2 * 1e3:+.2f} mrad",
            f"几何匹配        : m={res.get('m', res['sol'][3])} 环 | rms={res.get('rms', res['sol'][4]):.3f} px",
            f"Cake 1D 评分    : refined={sc['ps_refined']:.3f} | 命中峰={sc['n_peak_hit']} | |Δq|_med={dq_s}",
            f"置信度评级      : {conf.get('tier')} ({conf.get('badge')})",
            f"评定说明        : {conf.get('rationale')}",
            f"使用建议        : {conf.get('recommendation')}",
        ]
        self.txt_qa.setPlainText("\n".join(qa_lines))

        self.b_exp_poni.setEnabled(True)
        self.b_exp_fig.setEnabled(True)
        self.b_copy_txt.setEnabled(True)
        self.select_stage(3)
        self.tabs.setCurrentIndex(1)
        self.status_lbl.setText(f"全流程完成：{ph} | D={dist_m*1000:.3f} mm | {conf.get('tier')}")

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
            self.ax_2d.set_title(f"2D 环叠图 ({phase} | D={dist_m*1000:.2f} mm)", fontsize=10)
        else:
            self.ax_2d.set_title("2D 原始衍射图（红色半透明=掩膜区）", fontsize=10)
        self.fig_2d.tight_layout()
        self.cv_2d.draw_idle()

    def _draw_cake_tab(self, cake, qs, phase):
        draw_cake(self.ax_cake, cake, qs=qs, title=f"q-χ Cake 展开图 · {phase}（绿虚线为理论峰位）")
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
        self.ax_1d.set_title(f"1D 径向积分谱 · refined={sc['ps_refined']:.3f} | |Δq|_med={dq_s}", fontsize=10)
        self.fig_1d.tight_layout()
        self.cv_1d.draw_idle()

    def export_poni(self):
        if not self.payload:
            return
        base = os.path.splitext(os.path.basename(self.paths[0]))[0]
        p, _ = QFileDialog.getSaveFileName(self, "导出 .poni", f"{base}_calibrated.poni", "pyFAI PONI (*.poni)")
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
            self.cards[4].set_status("done", f"已导出 {os.path.basename(p)}")
            self.lbl_export.setText(f"已导出 .poni：\n{p}")
            self._log(f"[步骤5] 已导出 .poni -> {p}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def export_diag_fig(self):
        if not self.payload:
            return
        base = os.path.splitext(os.path.basename(self.paths[0]))[0]
        p, _ = QFileDialog.getSaveFileName(self, "导出标定体检图", f"{base}_体检图.png", "PNG (*.png);;PDF (*.pdf)")
        if not p:
            return
        try:
            export_diagnostic_figure(
                p, self.payload["img"], self.payload["res"],
                self.payload["px1_m"], px2_m=self.payload["px2_m"],
                mask=self.payload["mask"], spline=self.payload["spline"],
            )
            self.cards[4].set_status("done", f"已导出 {os.path.basename(p)}")
            self.lbl_export.setText(f"已导出体检图：\n{p}")
            self._log(f"[步骤5] 已导出体检图 -> {p}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def copy_summary(self):
        if not self.payload:
            return
        QApplication.clipboard().setText(self.txt_qa.toPlainText())
        self.status_lbl.setText("已将标定结果摘要复制到剪贴板。")

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
        self.lbl_file_info.setText("尚未导入文件")
        self.lbl_calib_mode.setText("求解模式：待导入图像后自动判断（单帧 / 多帧联合）")
        self.txt_qa.setPlainText("尚未运行标定。")
        self.txt_log.clear()
        self.b_run_all.setEnabled(False)
        self.b_exp_poni.setEnabled(False)
        self.b_exp_fig.setEnabled(False)
        self.b_copy_txt.setEnabled(False)
        for i, c in enumerate(self.cards):
            c.set_status("idle", STAGES[i][1])
        for ax, cv in ((self.ax_2d, self.cv_2d), (self.ax_cake, self.cv_cake), (self.ax_1d, self.cv_1d)):
            ax.clear()
            cv.draw_idle()
        self.select_stage(0)
        self.status_lbl.setText("已重置，请重新导入数据。")


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
        msg = (
            f"SELFTEST_OK phases={len(torchfai.phases.PHASE_NAMES)} "
            f"cards={len(win.cards)} pyFAI={getattr(pyFAI, 'version', 'ok')}"
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
