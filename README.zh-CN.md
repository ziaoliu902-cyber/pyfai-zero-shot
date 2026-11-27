<div align="center">

# pyfai-zero-shot

**无需先验物相与初始几何的二维粉晶 X 射线探测器自动标定引擎<br>全环与离轴四角残弧通用 · 11 语种桌面工作台**

🚫 无需指定标样物相 &nbsp;·&nbsp; 🚫 无需初始距离/束心 &nbsp;·&nbsp; 🌍 11 语言桌面端（含阿拉伯语 RTL）<br>✅ 直接导出标准 pyFAI <code>.poni</code> &nbsp;·&nbsp; ✅ 定量可辨识度评级 &nbsp;·&nbsp; ✅ 冻结基线 <code>f608e67</code>

[English](README.md) · **简体中文** · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Français](README.fr.md) · [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Commit](https://img.shields.io/badge/frozen-f608e67-0072B2)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()
[![Manuscript](https://img.shields.io/badge/manuscript-Science_Advances_Research_Resource-D55E00)](manuscript/pyFAI-ZeroShot_SciAdv_ResearchResource.pdf)

</div>

---

## 论文预印本（Science Advances — Research Resource 格式）

- 📄 **英文正文 PDF（10 页）**：[pyFAI-ZeroShot_SciAdv_ResearchResource.pdf](manuscript/pyFAI-ZeroShot_SciAdv_ResearchResource.pdf)（[Markdown](manuscript/SciAdv_ResearchResource_v1.md)）
- 📊 **补充材料 PDF（Tables S1–S5）**：[pyFAI-ZeroShot_SciAdv_Supplementary.pdf](manuscript/pyFAI-ZeroShot_SciAdv_Supplementary.pdf)（[Markdown](manuscript/SciAdv_Supplementary_v1.md)）
- 🔒 **全量冻结基准表（`f608e67`）**：[docs/paper_numbers_freeze_2026-11-27_f608e67.md](docs/paper_numbers_freeze_2026-11-27_f608e67.md)

## 多语言桌面工作台（11 种语言）

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — 5 阶段可视化流程图工作台，集成自动掩膜、Cake 体检与置信度红绿灯，内置 **`probes/i18n_wf.py` 11 语种实时切换**（简体中文、English、正體中文、日本語、Français、Deutsch、Русский、Português、한국어、Español、العربية 含原生 RTL 布局）。
- **DiffBlindWizard (`probes/gui_wizard.py`)** — 面向日常实验人员的 5 步向导式 `.poni` 生成工具。

## 核心原理

1. **自适应预处理**：`auto_offset` 自动扣除平板探测器负基底（`< -5% p99.9`）；`auto_mask` 自动屏蔽模块缝隙、坏点热像素与束流挡盘阴影。
2. **多尺度画布阶梯（`R2`）**：默认从 `≤ 520 px` 起跑，遇到无解、低环高残差或高降采样分叉风险（`s ≥ 5` 且 `|ΔD|/D ≥ 10%`）时自动升级至 `ceil(maxdim/2)`。
3. **零先验定心与物相仲裁**：Layer-0 谱锐度/Kåsa 守卫、环饥饿守卫（`|ln(D_RS/D_geo)| > 0.46`）、径向对称+射线多种子共识投票、全谱 Pearson 相关 `ps` 与覆盖分仲裁。
4. **全环与部分弧精修**：完整环走原生 2D Taubin 圆拟合；离轴四角残弧（`az_cov < 50%`）启用 `1440²` 正则化角束倾角与**两参数差分 Pseudo-Voigt 解耦**（`r_i = D tan(2θ_i) + δr_0` 并沿可见弧法向修正束心）。

<div align="center">
<img src="manuscript/figures/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and DESY 4-stage ablation" width="840">
</div>

## 冻结基线实测结果（`f608e67` 冷启动）

| 基准测试集 | 实测表现（`f608e67`，逐帧清缓存冷启动） |
| :--- | :--- |
| **DESY PETRA III P02.1 盲测**（13 组，230–2290 mm，Varex 2880² + PerkinElmer 2048²，~60 keV） | **13/13 (100%)** 命中 `LaB6`，**13/13 `< 0.12%`**，中位 **`0.030%`**（Locked OOD `0.028%`），R² `0.999999`，RMSE `0.82 mm`，角束定心 **`0.74 px`**（提升 `9.7×`），Mode B `8/8` 逐位一致 |
| **CLS 08B1-1 真实矿物**（16 帧，4 种含砷矿物，COD/pymatgen 自动生成相表） | **16/16 (100%)** 直接收敛（0 个假盆地），**`246.90 ± 1.36 mm`**，MAD **`0.69 mm`**，`ps = +0.583..+0.879`，中位耗时 `12.96 s` |
| **ESRF D2AM `MX_ceria` & `Al2O3`**（7 档 150–450 mm；52 帧含 1 暗场） | `CeO2` 纯盲解 **7/7**（均差 `0.139 mm`）/ 锁表 `phase_fit` 均差与最差 **`0.1 mm`**；`Al2O3` **51/52**（自动拒收暗场），`.spline` 均值 **`116.70 mm`** vs 真值 `116.86 mm`（`−0.16 mm`） |
| **跨探测器 `silx_zoo`**（已知 λ，盲物相） | **5/5** 物相全中：`Pilatus2M −0.007%`、`Eiger4M +0.041%`、`Eiger4M_v2 +0.311%`、`Eiger9M +0.008%`、`Pilatus1M`（`AgBh` 物理量 λ·D 误差 `+0.14%`） |
| **100 例合成可辨识性边界**（`Seed = 20261120`） | 清晰相变位于 `n_vis ≥ 4`：**41/41 (100%) `≤ 1.0%`**，**40/41 (97.6%) `≤ 0.20%`**，中位 **`0.0010%`**（`nvis = 5–13`：`31/31 ≤ 0.0019%`）；`n_vis ≤ 3` 仅 `27.1%` |
| **与无加权 `pyFAI` 24 帧对比** | 零初值 `pyFAI`（`P0`，给定标样）：仅 **3/24 (12.5%)**（DESY 角束 `0/8`，中位 `45.81%`）→ 本引擎（盲物相）：**24/24 (100%)**（`0.033%`）→ 用本引擎初值驱动 `pyFAI`（`P1`）：**24/24 (100%)**，中位 **`0.006%` / `0.11 px`** |

<details>
<summary><b>更多论文主图（DESY 盲测 · 可辨识性相变 · 跨光源实测 · pyFAI 对比）</b></summary>

<div align="center">
<img src="manuscript/figures/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="manuscript/figures/fig3_identifiability_bound.png" width="800"><br><br>
<img src="manuscript/figures/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="manuscript/figures/fig5_pyfai_comparison.png" width="820">
</div>

</details>

## 快速开始

```bash
pip install numpy scipy matplotlib pyFAI fabio pillow h5py PyQt5
```

```python
import fabio, numpy as np
from probes.solver_core import auto_offset, solve_image
from probes.auto_mask import build_auto_mask

img = auto_offset(fabio.open("your_frame.tif").data.astype(np.float64))
mask, _ = build_auto_mask(img)
res = solve_image(img, px_m=150e-6, lam=0.20735, phases=None, mask=mask)
print(res["phase"], res["sol"], res["confidence"]["tier"])
```

```bash
# Launch the 11-language desktop workflow GUI (zh/en/zh_TW/ja/fr/de/ru/pt/ko/es/ar)
python probes/gui_workflow.py
# Or launch the 5-step beginner wizard
python probes/gui_wizard.py
```

## 引用方式

```bibtex
@article{liu2026pyfaizeroshot,
  author  = {Liu, Zi'ao and Liu, Jiaru and Liu, Haixia},
  title   = {Phase-blind, zero-prior geometric calibration of two-dimensional powder X-ray diffraction detectors across full rings and off-axis partial arcs},
  year    = {2026},
  note    = {Science Advances Research Resource manuscript; frozen release commit f608e67},
  url     = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot}
}
```

## 开源协议与致谢

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for acquiring and providing the NIST SRM 660c LaB6 flat-panel calibration datasets.

<sub>MIT 开源协议（见 [LICENSE](LICENSE)）。所有多语言 README 已与 `f608e67` 同步更新。</sub>
