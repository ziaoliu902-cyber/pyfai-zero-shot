<div align="center">

# pyfai-zero-shot

**Слепая геометрическая калибровка 2D-детекторов порошковой дифракции без начальных приближений<br>Для полных колец и внеосевых угловых дуг · Экспорт .poni для pyFAI**

🚫 Без указания эталона &nbsp;·&nbsp; 🚫 Без начальной геометрии &nbsp;·&nbsp; 🌍 GUI на 11 языках (вкл. арабский RTL)<br>✅ Экспорт <code>.poni</code> pyFAI &nbsp;·&nbsp; ✅ Cake-диагностика &nbsp;·&nbsp; ✅ Загрузка своих CIF

[English](README.md) · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · **Русский** · [Français](README.fr.md) · [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/built%20for-pyFAI-013243)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()

</div>

---

## Зачем нужен pyfai-zero-shot?

Стандартные пакеты (`pyFAI-calib2`, `DIOPTAS`, `DAWN`) дают субпиксельную точность при хорошем начальном приближении, но легко застревают в ложных минимумах без заданной дистанции или при смещении пучка в угол детектора.

`pyfai-zero-shot` восстанавливает фазу и геометрию из одного изображения, размера пикселя и длины волны и передаёт готовый `.poni` в `pyFAI`.

## Настольный интерфейс (11 языков)

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — 5-этапный визуальный интерфейс с переключением **11 языков** (`probes/i18n_wf.py`: zh, en, zh_TW, ja, fr, de, ru, pt, ko, es, ar с RTL).
- **DiffBlindWizard (`probes/gui_wizard.py`)** — пошаговый мастер калибровки.
- **Свои фазы (`probes/custom_phase.py`)** — импорт любых файлов `.cif` или `.D`.

## Принцип работы

1. **Предобработка:** `auto_offset` и `auto_mask` убирают отрицательный пьедестал, зазоры модулей, битые пиксели и тень бимстопа.
2. **Многомасштабная лестница (`R2`):** Старт с `≤ 520 px` и авто-переход к `ceil(maxdim/2)` при расхождении.
3. **Поиск центра и фазы:** Контроль резкости/Kåsa, радиальная симметрия + лучевые семена, консенсусное голосование и корреляция профиля `ps`.
4. **Уточнение:** 2D-аппроксимация Taubin для полных колец; регуляризованный наклон `1440²` + **2-параметрическая модель Pseudo-Voigt** для угловых дуг.

<div align="center">
<img src="assets/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and ablation" width="840">
</div>

## Точность на реальных и синтетических данных

| Набор данных | Точность (холодный старт) |
| :--- | :--- |
| **DESY PETRA III P02.1** (13 сканов, 230–2290 мм) | **13/13 (100%)** `LaB6`, **13/13 `< 0.12%`**, медиана **`0.030%`**, R² `0.999999`, угловой ΔC **`0.74 px`** |
| **CLS 08B1-1 Минералы** (16 кадров, 4 минерала) | **16/16 (100%)** сходимость, **`246.90 ± 1.36 мм`**, MAD **`0.69 мм`** |
| **ESRF D2AM `MX_ceria` и `Al2O3`** | `CeO2` слепо **7/7** (`0.139 мм`); `Al2O3` **51/52**, `.spline` **`116.70 мм`** против GT `116.86 мм` |
| **Кросс-детекторный `silx_zoo`** | **5/5** фаз (`Pilatus2M −0.007%`, `Eiger4M +0.041%`, `Eiger9M +0.008%`) |
| **100 синтетических кадров** | При **≥ 4 кольцах**: **41/41 `≤ 1.0%`**, **40/41 `≤ 0.20%`**, медиана **`0.0010%`** (≤ 3: `27.1%`) |
| **Сравнение с `pyFAI`** (24 кадра) | `pyFAI` без нач. геом. **3/24 (12.5%)** → `pyFAI-ZeroShot` **24/24 (100%)** (`0.033%`) → `pyFAI` от нашего результата **24/24 (`0.006%` / `0.11 px`)** |

<details>
<summary><b>Показать дополнительные графики точности</b></summary>

<div align="center">
<img src="assets/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="assets/fig3_identifiability_bound.png" width="800"><br><br>
<img src="assets/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="assets/fig5_pyfai_comparison.png" width="820">
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

## Цитирование

```bibtex
@software{pyfai_zero_shot_2026,
  author  = {Liu, Zi'ao and Liu, Jiaru and Liu, Haixia},
  title   = {pyfai-zero-shot: phase-blind, zero-prior geometry calibration of 2-D X-ray powder diffraction detectors},
  year    = {2026},
  url     = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot},
  license = {MIT}
}
```

## Лицензия и благодарности

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for providing the NIST SRM 660c LaB6 flat-panel calibration datasets used in our benchmark validation, as well as the pyFAI community, ESRF D2AM (BM02), and CLS 08B1-1 for open diffraction datasets.

<sub>Лицензия MIT ([LICENSE](LICENSE)).</sub>
