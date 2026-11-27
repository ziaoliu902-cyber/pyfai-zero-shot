<div align="center">

# pyfai-zero-shot

**Слепая геометрическая калибровка 2D-детекторов порошковой рентгеновской дифракции без априорных данных<br>для полных колец и внеосевых частичных дуг**

🚫 Без указания эталона &nbsp;·&nbsp; 🚫 Без начальной геометрии &nbsp;·&nbsp; 🌍 GUI на 11 языках (вкл. арабский RTL)<br>✅ Экспорт <code>.poni</code> pyFAI &nbsp;·&nbsp; ✅ Уровни идентифицируемости &nbsp;·&nbsp; ✅ Коммит <code>f608e67</code>

[English](README.md) · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · **Русский** · [Français](README.fr.md) · [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Commit](https://img.shields.io/badge/frozen-f608e67-0072B2)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()
[![Manuscript](https://img.shields.io/badge/manuscript-Science_Advances_Research_Resource-D55E00)](manuscript/pyFAI-ZeroShot_SciAdv_ResearchResource.pdf)

</div>

---

## Рукопись и препринт (Science Advances — Research Resource)

- 📄 **Основная рукопись PDF (10 стр.):** [pyFAI-ZeroShot_SciAdv_ResearchResource.pdf](manuscript/pyFAI-ZeroShot_SciAdv_ResearchResource.pdf)
- 📊 **Дополнительные материалы PDF (Таблицы S1–S5):** [pyFAI-ZeroShot_SciAdv_Supplementary.pdf](manuscript/pyFAI-ZeroShot_SciAdv_Supplementary.pdf)
- 🔒 **Зафиксированные бенчмарки (`f608e67`):** [docs/paper_numbers_freeze_2026-11-27_f608e67.md](docs/paper_numbers_freeze_2026-11-27_f608e67.md)

## Настольный GUI (11 языков)

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — 5-этапный визуальный интерфейс с авто-маской, Cake-диагностикой и переключением **11 языков** (`probes/i18n_wf.py`: zh, en, zh_TW, ja, fr, de, ru, pt, ko, es, ar с RTL).
- **DiffBlindWizard (`probes/gui_wizard.py`)** — пошаговый мастер калибровки.

## Принцип работы

1. **Предобработка:** `auto_offset` удаляет отрицательный пьедестал плоских панелей; `auto_mask` маскирует зазоры модулей, битые пиксели и тень бимстопа.
2. **Многомасштабная лестница (`R2`):** Старт с `≤ 520 px`, автоматический переход к `ceil(maxdim/2)` при дивергенции (`s ≥ 5`, `|ΔD|/D ≥ 10%`).
3. **Центрирование и арбитраж фаз:** Контроль резкости/Kåsa, радиальная симметрия + лучевые семена, консенсусное голосование и корреляция профиля `ps`.
4. **Уточнение:** 2D-аппроксимация Taubin для полных колец; регуляризованный наклон `1440²` + **2-параметрическая дифференциальная модель Pseudo-Voigt** (`r_i = D tan(2θ_i) + δr_0`) для угловых дуг (`az_cov < 50%`).

<div align="center">
<img src="manuscript/figures/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and DESY 4-stage ablation" width="840">
</div>

## Проверенные бенчмарки (`f608e67`, холодный старт)

| Набор данных | Результат (`f608e67`) |
| :--- | :--- |
| **DESY PETRA III P02.1** (13 сканов, 230–2290 мм) | **13/13 (100%)** `LaB6`, **13/13 `< 0.12%`**, медиана **`0.030%`** (OOD `0.028%`), R² `0.999999`, RMSE `0.82 мм`, угловой ΔC **`0.74 px`** (`9.7×`) |
| **CLS 08B1-1 Минералы** (16 кадров, 4 минерала мышьяка) | **16/16 (100%)** прямая сходимость, **`246.90 ± 1.36 мм`**, MAD **`0.69 мм`** |
| **ESRF D2AM `MX_ceria` и `Al2O3`** | `CeO2` слепо **7/7** (`0.139 мм`) / `phase_fit` **`0.1 мм`**; `Al2O3` **51/52**, `.spline` **`116.70 мм`** против GT `116.86 мм` |
| **Кросс-детекторный `silx_zoo`** | **5/5** фаз (`Pilatus2M −0.007%`, `Eiger4M +0.041%`, `Eiger9M +0.008%`) |
| **100 синтетических кадров** (`Seed = 20261120`) | Чёткий порог `n_vis ≥ 4`: **41/41 `≤ 1.0%`**, **40/41 `≤ 0.20%`**, медиана **`0.0010%`** против `n_vis ≤ 3` (`27.1%`) |
| **Сравнение с `pyFAI`** (24 реальных кадра) | `pyFAI` без нач. геом. **3/24 (12.5%)** → `Ours` **24/24 (100%)** (`0.033%`) → `pyFAI` с нашей инициализацией **24/24 (100%)** (**`0.006%` / `0.11 px`**) |

<details>
<summary><b>Дополнительные рисунки из статьи</b></summary>

<div align="center">
<img src="manuscript/figures/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="manuscript/figures/fig3_identifiability_bound.png" width="800"><br><br>
<img src="manuscript/figures/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="manuscript/figures/fig5_pyfai_comparison.png" width="820">
</div>

</details>

## Быстрый старт

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

## Цитирование

```bibtex
@article{liu2026pyfaizeroshot,
  author  = {Liu, Zi'ao and Liu, Jiaru and Liu, Haixia},
  title   = {Phase-blind, zero-prior geometric calibration of two-dimensional powder X-ray diffraction detectors across full rings and off-axis partial arcs},
  year    = {2026},
  note    = {Science Advances Research Resource manuscript; frozen release commit f608e67},
  url     = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot}
}
```

## Лицензия и благодарности

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for acquiring and providing the NIST SRM 660c LaB6 flat-panel calibration datasets.

<sub>Лицензия MIT ([LICENSE](LICENSE)). Синхронизировано с коммитом `f608e67`.</sub>
