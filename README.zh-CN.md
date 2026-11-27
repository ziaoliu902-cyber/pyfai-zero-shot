<div align="center">

# pyfai-zero-shot

**无需先验物相与初始几何的二维粉晶 X 射线探测器自动标定工具<br>全环与离轴四角残弧通用 · 开箱生成标准 pyFAI <code>.poni</code>**

🚫 无需指定标样名称 &nbsp;·&nbsp; 🚫 无需初始距离/束心猜测 &nbsp;·&nbsp; 🌍 11 语言桌面界面（含阿拉伯语 RTL）<br>✅ 一键导出标准 pyFAI <code>.poni</code> &nbsp;·&nbsp; ✅ 内置 Cake 展开体检与置信度评级 &nbsp;·&nbsp; ✅ 支持自定义 <code>.cif</code> 物相

[English](README.md) · **简体中文** · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Français](README.fr.md) · [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/built%20for-pyFAI-013243)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()

</div>

---

## 为什么做这个工具？

日常做二维衍射实验时，`pyFAI-calib2`、`DIOPTAS` 等软件在初值准确时精度极高，但在**未知初始距离、束心打偏到探测器角落、或者不同距离（`0.2–2.3 m`）之间频繁切换**时，常规自动算法极易卡进错位局部极小或需要人工点环。

`pyfai-zero-shot` 只需要**一张衍射图 + 像素尺寸 + 波长**即可全自动完成物相识别与几何标定，输出标准 `.poni` 文件（也可直接给 `pyFAI` 做二段亚像素精修）。

## 11 语言桌面工作台与向导

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — 5 阶段可视化流程图工作台（数据导入 → 自动掩膜/畸变 → 零先验标定 → Cake 均匀度体检 → 一键导出 `.poni` 与 2×2 诊断图），支持 **`probes/i18n_wf.py` 11 语种实时切换**（简体中文、English、正體中文、日本語、Français、Deutsch、Русский、Português、한국어、Español、العربية 含原生 RTL 布局）。
- **DiffBlindWizard (`probes/gui_wizard.py`)** — 5 步向导式界面，带红绿灯置信度提示，方便实验人员直接上手。
- **自定义物相 (`probes/custom_phase.py`)** — 除了内置 `LaB6` (SRM 660c)、`CeO2`、`Al2O3`、`Si`、`Quartz`、`AgBh`、`AgSt` 外，支持直接导入任意 `.cif` 或 `.D` 文件。

## 工作原理

1. **自适应预处理**：`auto_offset` 自动扣除平板探测器负基底（`< -5% p99.9`）；`auto_mask` 自动屏蔽模块缝隙、坏点热像素与束挡阴影。
2. **多尺度画布阶梯 (`R2`)**：默认在 `≤ 520 px` 画布计算，当内环密集发生降采样合并时（`s ≥ 5` 且 `|ΔD|/D ≥ 10%`）自动升级到 `ceil(maxdim/2)` 分辨率重解。
3. **零先验定心与物相仲裁**：结合谱锐度/Kåsa 圆心守卫、径向对称 + 射线多种子共识投票，并通过全谱 Pearson 相关 `ps` 锁定最优物相。
4. **全环与离轴部分弧精修**：完整环采用原生 2D Taubin 圆拟合；针对方位覆盖率 `< 50%` 的角束残弧，启用 `1440²` 正则化倾角与**两参数差分 Pseudo-Voigt 解耦**（`r_i = D tan(2θ_i) + δr_0` 并沿可见弧法向修正束心）。

<div align="center">
<img src="assets/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and ablation" width="840">
</div>

## 跨光源与跨探测器实测表现（冷启动、无初始几何）

| 测试集 | 实测精度与表现 |
| :--- | :--- |
| **DESY PETRA III P02.1**（13 组，230–2290 mm，Varex 2880² + PerkinElmer 2048²，~60 keV） | **13/13 (100%)** 识别 `LaB6`，**13/13 `< 0.12%`**，中位误差 **`0.030%`**，R² `0.999999`，离轴角束束心中位 **`0.74 px`**（中位 `26.6 s/帧`） |
| **CLS 08B1-1 真实矿物**（16 帧，4 种含砷矿物，COD CIF 自动导表） | **16/16 (100%)** 直接收敛（`scorodite` / `realgar` / `orpiment` / `arsenopyrite` 统一 **`246.90 ± 1.36 mm`**，MAD **`0.69 mm`**，中位 `12.96 s`） |
| **ESRF D2AM `MX_ceria` & `Al2O3`**（150–450 mm；52 帧含 1 张暗场） | `CeO2` 盲物相 **7/7**（均差 `0.139 mm`）/ 锁表 2D **`0.1 mm`**；`Al2O3` **51/52**（暗场自动拒收），`.spline` 均值 **`116.70 mm`** vs 真值 `116.86 mm` |
| **跨探测器测试 (`silx_zoo`)** | **5/5** 物相全中：`Pilatus2M −0.007%`、`Eiger4M +0.041%`、`Eiger4M_v2 +0.311%`、`Eiger9M +0.008%`、`Pilatus1M`（`AgBh` 物理量 λ·D `+0.14%`） |
| **100 例合成随机扫参边界** | 当可见环 **≥ 4 条**：**41/41 (100%) `≤ 1.0%`**，**40/41 (97.6%) `≤ 0.20%`**，中位 **`0.0010%`**（`5–13` 环全部 `≤ 0.0019%`）；`≤ 3` 环进入物理欠定区（仅 `27.1%`），软件会主动给出风险提示 |
| **与无加权 `pyFAI` 24 帧实测对比** | `pyFAI` 零初值直接跑（已知标样）：仅 **3/24 (12.5%)**（角束 `0/8`）→ 本工具（盲物相）：**24/24 (100%)**（`0.033%`）→ **用本工具结果给 `pyFAI` 当初值**：**24/24 (100%)** 收敛至 **`0.006%` / `0.11 px`** |

<details>
<summary><b>展开更多测试效果图（DESY 盲测对比 · 环数可辨识边界 · 多光源测试 · 与 pyFAI 对比）</b></summary>

<div align="center">
<img src="assets/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="assets/fig3_identifiability_bound.png" width="800"><br><br>
<img src="assets/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="assets/fig5_pyfai_comparison.png" width="820">
</div>

</details>

## 快速上手

```bash
pip install numpy scipy matplotlib pyFAI fabio pillow h5py PyQt5
```

```python
import fabio, numpy as np
from probes.solver_core import auto_offset, solve_image
from probes.auto_mask import build_auto_mask
from probes.cake_report import export_poni_from_result

# Load any 2-D powder diffraction image (TIFF / EDF / CBF / HDF5)
img = auto_offset(fabio.open("your_frame.tif").data.astype(np.float64))
mask, _ = build_auto_mask(img)

# Zero-prior solve: provide only pixel pitch (m) and wavelength (Å)
res = solve_image(img, px_m=150e-6, lam=0.20735, phases=None, mask=mask)
print(f"Phase: {res['phase']}, m={res['m']}, ps={res['ps']:.3f}, tier={res['confidence']['tier']}")
```

```bash
# 11-language visual workflow workbench (zh/en/zh_TW/ja/fr/de/ru/pt/ko/es/ar incl. RTL)
python probes/gui_workflow.py

# 5-step guided calibration wizard for beginners
python probes/gui_wizard.py
```

## 引用本项目

```bibtex
@software{pyfai_zero_shot_2026,
  author  = {Liu, Zi'ao and Liu, Jiaru and Liu, Haixia},
  title   = {pyfai-zero-shot: phase-blind, zero-prior geometry calibration of 2-D X-ray powder diffraction detectors},
  year    = {2026},
  url     = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot},
  license = {MIT}
}
```

## 开源协议与致谢

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for providing the NIST SRM 660c LaB6 flat-panel calibration datasets used in our benchmark validation, as well as the pyFAI community, ESRF D2AM (BM02), and CLS 08B1-1 for open diffraction datasets.

<sub>MIT 开源协议（见 [LICENSE](LICENSE)）。本工具可无缝对接 pyFAI 使用。</sub>
