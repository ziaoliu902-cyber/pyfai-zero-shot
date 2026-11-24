# -*- coding: utf-8 -*-
"""confidence_map.py：二维置信度评定与可视化模块 (m vs ps)

从单指标 Pattern Score 升级为 (匹配环数 m, 模式匹配度 ps) 的二维置信度空间。
理论依据：
  1. 可辨识性数学界限（M6 / R4 论文定理）：
     - m >= 4：反问题良定，100% 恢复几何且误差 < 1%（中位数 0.00%）；
     - m <= 3：反问题欠定简并（D 与束心平移代偿），成功率仅 27%。
  2. 模式评分 ps（Pearson 关联度）：
     - ps >= 0.45：正相真盆地（CLS 矿物基准实证：正相 ps > 0.45，错相 ps < 0.15）；
     - 0.20 <= ps < 0.40：过渡区（弱环、择优取向织构或谐波标样）；
     - ps < 0.20：错误物相或未收敛假解。
  3. 方位角覆盖率 az_cov：
     - az_cov < 0.40：部分弧偏差风险（即使 m>=4，单帧亦可能落入假解）。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.figure import Figure

# 中文字体兼容
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                          "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def evaluate_confidence(m, ps, max_dev=None, az_cov=1.0, cv_ratio=None):
    """计算当前解在二维置信度空间中的评级与决策建议。

    参数：
      m: 匹配环数 (int)
      ps: 模式匹配度 Pattern Score (float, [0, 1])
      max_dev: 最大 2θ 偏差（度），自检残差
      az_cov: 方位角覆盖率 (float, [0, 1])，默认 1.0 (全环)
      cv_ratio: 拆分交叉验证残差比 (m <= 5 时生效)

    返回：
      dict(tier, badge, color, is_identifiable, rationale, recommendation)
    """
    m = int(m) if m is not None else 0
    ps = float(ps) if ps is not None and np.isfinite(ps) else 0.0
    az_cov = float(az_cov) if az_cov is not None and np.isfinite(az_cov) else 1.0

    # 1. 基础物理可辨识性判定
    is_identifiable = (m >= 4) and (az_cov >= 0.40)

    # 2. 细分等级
    if az_cov < 0.40:
        tier = "PARTIAL_ARC"
        badge = "⚠️ 部分弧警示 (方位角 < 40%)"
        color = "#e67e22"  # 橙色
        rationale = f"检测到部分残弧（方位覆盖仅 {az_cov:.0%}），存在束心-距离代偿的客观假解风险。"
        recommendation = "单帧存在数学不可辨识性；建议提供近似距离先验，或启用多帧方位角堆叠功能。"
    elif m < 4:
        tier = "LOW"
        badge = "🔴 欠定简并 (m <= 3 环)"
        color = "#e74c3c"  # 红色
        rationale = f"可见环仅 {m} 条，位于反问题数学欠定域（基准统计成功率仅 27%）。"
        recommendation = "环数过少无法独立约束 5 参数；建议提供波长/近似距离先验，或切换更大画布尝试找弱环。"
    elif ps < 0.20:
        tier = "LOW"
        badge = "🔴 匹配不佳 (ps < 0.20)"
        color = "#c0392b"  # 深红
        rationale = f"Pattern Score ({ps:.3f}) 显著偏低，可能匹配了错误的晶体相。"
        recommendation = "建议核实相库（尝试 CeO2/LaB6/Si/Al2O3 等）或检查波长输入是否吻合。"
    elif cv_ratio is not None and np.isfinite(cv_ratio) and m <= 5 and cv_ratio > 2.5:
        tier = "LOW"
        badge = "🔴 少环过拟合风险 (cv_ratio > 2.5)"
        color = "#e74c3c"
        is_identifiable = False
        rationale = f"环数较少 (m={m} <= 5) 且拆分交叉验证外推残差比 cv_ratio={cv_ratio:.2f} > 2.5，疑似落入伪盆地。"
        recommendation = "建议核对物相与波长、增加近似距离先验，或换更大画布检测更多衍射环。"
    elif ps >= 0.40 and (max_dev is None or max_dev <= 0.05):
        tier = "HIGH"
        badge = "🟢 极高置信 (良定且吻合)"
        color = "#27ae60"  # 绿色
        dev_str = f"，自检最大偏差 {max_dev:.3f}° <= 0.05°" if max_dev is not None else ""
        rationale = f"环数 m={m} >= 4 (良定域)，Pattern Score={ps:.3f} >= 0.40{dev_str}。"
        recommendation = "几何解高度稳健，基准重现精度 < 0.1%，可放心导出标准 .poni 标定文件。"
    else:
        tier = "MEDIUM"
        badge = "🟡 中等置信 (警示/需复核)"
        color = "#f39c12"  # 黄色
        rationale = f"环数 m={m} 进入良定域，但模式得分 ({ps:.3f}) 处于过渡带或自检存在微小偏差。"
        recommendation = "建议通过图形界面检查衍射环叠加，或启动 5 参数精修 (refine5) 进一步优化。"

    return dict(
        tier=tier,
        badge=badge,
        color=color,
        is_identifiable=is_identifiable,
        m=m,
        ps=ps,
        az_cov=az_cov,
        cv_ratio=cv_ratio,
        rationale=rationale,
        recommendation=recommendation
    )


def draw_confidence_map(ax, m=None, ps=None, az_cov=1.0, max_dev=None,
                        cv_ratio=None,
                        title="二维置信度空间 (m vs ps)", theme_name="light"):
    """在给定的 matplotlib Axes 上绘制 2D 置信度区域图及当前解的位置。"""
    is_dark = (theme_name == "dark")
    bg_color = "#1e1e2e" if is_dark else "#fafafa"
    text_color = "#cdd6f4" if is_dark else "#2c3e50"
    grid_color = "#45475a" if is_dark else "#e0e0e0"

    ax.clear()
    ax.set_facecolor(bg_color)

    # 绘制背景决策分区
    # 区域 1: 欠定危险区 (m <= 3.5, 0 <= ps <= 1.0)
    ax.add_patch(Rectangle((0, 0), 3.5, 1.0, color="#ff7675", alpha=0.35 if not is_dark else 0.25,
                           label="欠定区 (m<=3)"))
    # 区域 2: 错相低分危险区 (m >= 3.5, 0 <= ps <= 0.20)
    ax.add_patch(Rectangle((3.5, 0), 16.5, 0.20, color="#fab1a0", alpha=0.30 if not is_dark else 0.20,
                           label="假盆地 (ps<0.2)"))
    # 区域 3: 过渡警告区 (m >= 3.5, 0.20 <= ps <= 0.40)
    ax.add_patch(Rectangle((3.5, 0.20), 16.5, 0.20, color="#ffeaa7", alpha=0.35 if not is_dark else 0.25,
                           label="过渡区 (0.2<=ps<0.4)"))
    # 区域 4: 高置信区 (m >= 3.5, 0.40 <= ps <= 1.0)
    ax.add_patch(Rectangle((3.5, 0.40), 16.5, 0.60, color="#55efc4", alpha=0.35 if not is_dark else 0.25,
                           label="高置信区 (m>=4, ps>=0.4)"))

    # 边界线
    ax.axvline(3.5, color="#e74c3c", linestyle="--", lw=1.2, alpha=0.8)
    ax.text(3.6, 0.03, "m=4 理论界限", fontsize=7, color="#c0392b", rotation=90)

    ax.axhline(0.40, color="#27ae60", linestyle=":", lw=1.0, alpha=0.8)
    ax.text(14.5, 0.415, "ps=0.40 门控", fontsize=7, color="#27ae60")

    # 标记当前点
    if m is not None and ps is not None:
        eval_res = evaluate_confidence(m, ps, max_dev=max_dev, az_cov=az_cov,
                                       cv_ratio=cv_ratio)
        pt_color = eval_res["color"]
        marker = "X" if eval_res["tier"] in ("LOW", "PARTIAL_ARC") else "o"
        ax.plot(m, ps, marker=marker, markersize=10, color=pt_color,
                markeredgecolor="black" if not is_dark else "white", markeredgewidth=1.5,
                zorder=10)

        # 标注文本（去 emoji 防 matplotlib 字体缺失警告）
        clean_badge = eval_res['badge'].replace("🟢", "").replace("🟡", "").replace("🔴", "").replace("⚠️", "").strip()
        ann_text = f"{clean_badge}\n(m={m}, ps={ps:.2f})"
        if cv_ratio is not None and np.isfinite(cv_ratio):
            ann_text += f"\ncv={cv_ratio:.2f}"
        ax.annotate(ann_text, xy=(m, ps),
                    xytext=(m + (1.2 if m < 14 else -5.5), ps + (-0.12 if ps > 0.7 else 0.08)),
                    fontsize=8, fontweight="bold", color=text_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color,
                              edgecolor=pt_color, lw=1.2, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=pt_color, lw=1.0),
                    zorder=11)

    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("匹配环数 m", fontsize=8, color=text_color)
    ax.set_ylabel("Pattern Score (ps)", fontsize=8, color=text_color)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", color=text_color)
    ax.grid(True, linestyle="--", alpha=0.4, color=grid_color)
    ax.tick_params(colors=text_color, labelsize=7)


def create_confidence_figure(m=None, ps=None, az_cov=1.0, max_dev=None,
                             cv_ratio=None,
                             figsize=(3.4, 2.6), dpi=100, theme_name="light"):
    """创建并返回包含二维置信度空间的独立 matplotlib Figure。"""
    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    draw_confidence_map(ax, m=m, ps=ps, az_cov=az_cov, max_dev=max_dev,
                        cv_ratio=cv_ratio,
                        theme_name=theme_name)
    fig.tight_layout()
    return fig
