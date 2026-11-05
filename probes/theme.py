# -*- coding: utf-8 -*-
"""theme.py：科研桌面 UI 设计令牌与主题（sci-desktop-ui 规范落地，2026-11-02）。

 单一强调色 / 6px 圆角 / 8px 网格；亮暗双主题一套令牌生成 QSS；
 matplotlib 画布跟随主题；数字表格等宽字体；绘图用 Okabe-Ito 色盲
 安全色环。 PyQt5 先行（PyQt6 同构，届时只换 import）。
"""
import matplotlib as mpl

# ---- 设计令牌（spec §2）----
TOKENS = {
    "light": dict(bg="#FAFAF9", surface="#FFFFFF", border="#D9DCE0",
                  text="#1F2328", muted="#6B7280", accent="#2563EB",
                  danger="#C2410C", good="#0A7D0A"),
    "dark": dict(bg="#1E2124", surface="#272B30", border="#3A3F45",
                 text="#E6E8EB", muted="#9AA1AA", accent="#5B9BFF",
                 danger="#F08A5D", good="#5BC85B"),
}

# Okabe-Ito 色盲安全环（spec §4）：数据/拟合/标记统一从这里取
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
         "#E69F00", "#56B4E9", "#F0E442", "#000000"]
C_DATA, C_PEAK, C_PRED = OKABE[0], OKABE[1], OKABE[2]   # 谱线/检测峰/预测环
MONO = "Consolas"


def qss(name):
    """按令牌生成应用级 QSS（QApplication.setStyleSheet 用）。"""
    t = TOKENS[name]
    return f"""
* {{ font-family: "Segoe UI", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
     font-size: 10.5pt; }}
QMainWindow, QWizard, QDialog {{ background: {t['bg']}; }}
QGroupBox {{ background: {t['surface']}; border: 1px solid {t['border']};
             border-radius: 6px; margin-top: 10px; padding: 8px 8px 8px 8px;
             font-weight: 600; color: {t['text']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: {t['muted']}; }}
QPushButton {{ background: {t['surface']}; color: {t['text']};
               border: 1px solid {t['border']}; border-radius: 6px;
               padding: 5px 12px; }}
QPushButton:hover {{ border-color: {t['accent']}; }}
QPushButton:focus {{ border: 1px solid {t['accent']}; }}
QPushButton:disabled {{ color: {t['muted']}; border-color: {t['border']}; }}
QPushButton[primary="true"] {{ background: {t['accent']};
                               color: #FFFFFF; border: none;
                               font-weight: 600; }}
QPushButton[primary="true"]:disabled {{ background: {t['border']};
                               color: {t['muted']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 6px; padding: 4px 8px;
    selection-background-color: {t['accent']}; }}
QLineEdit:focus {{ border: 1px solid {t['accent']}; }}
QTextEdit, QPlainTextEdit {{ background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 6px;
    font-family: "{MONO}", monospace; font-size: 9.5pt; }}
QProgressBar {{ border: 1px solid {t['border']}; border-radius: 6px;
    background: {t['surface']}; text-align: center; color: {t['text']};
    min-height: 14px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 5px; }}
QLabel {{ color: {t['text']}; background: transparent; }}
QLabel[muted="true"] {{ color: {t['muted']}; font-weight: 400; }}
QStatusBar {{ background: {t['bg']}; color: {t['muted']}; }}
QToolBar {{ background: {t['bg']}; border: none; spacing: 4px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 4px;
                               min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QTabBar::tab {{ padding: 4px 14px; color: {t['muted']}; }}
QTabBar::tab:selected {{ color: {t['accent']}; border-bottom: 2px solid {t['accent']}; }}
QToolTip {{ background: {t['surface']}; color: {t['text']};
            border: 1px solid {t['border']}; padding: 4px; }}
QWizard QLabel {{ color: {t['text']}; }}
QWizard QWizardPage {{ background: {t['bg']}; }}
QCheckBox {{ color: {t['text']}; }}
"""


def apply_app(app, name):
    app.setStyleSheet(qss(name))


def mpl_theme(name):
    """matplotlib rcParams 快照（画布跟随主题；图像区保持灰图不变色）。"""
    t = TOKENS[name]
    return {
        "figure.facecolor": t["bg"], "axes.facecolor": t["surface"],
        "axes.edgecolor": t["border"], "axes.labelcolor": t["text"],
        "text.color": t["text"], "xtick.color": t["muted"],
        "ytick.color": t["muted"], "grid.color": t["border"],
        "savefig.facecolor": t["bg"],
    }


def restyle_axes(ax, name):
    """逐轴重刷主题（切换主题时对已存在的画布调用）。"""
    t = TOKENS[name]
    ax.set_facecolor(t["surface"])
    ax.figure.patch.set_facecolor(t["bg"])
    ax.tick_params(colors=t["muted"], direction="in")
    for s in ax.spines.values():
        s.set_color(t["border"])
    if ax.title:
        ax.title.set_color(t["text"])
    for lb in (ax.xaxis.label, ax.yaxis.label):
        lb.set_color(t["text"])


def use_mpl(name):
    mpl.rcParams.update(mpl_theme(name))
