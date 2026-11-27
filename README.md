<div align="center">

# pyfai-zero-shot

**Phase-blind, zero-prior geometric calibration of 2-D X-ray diffraction detectors<br>Works on full rings and off-axis corner arcs · Ready for pyFAI**

🚫 No calibrant name needed &nbsp;·&nbsp; 🚫 No initial distance/center guess &nbsp;·&nbsp; 🌍 11-language GUI (incl. RTL Arabic)<br>✅ Standard pyFAI <code>.poni</code> export &nbsp;·&nbsp; ✅ Built-in Cake QA &amp; reliability tiers &nbsp;·&nbsp; ✅ Full custom CIF support

**English** · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Français](README.fr.md) · [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/built%20for-pyFAI-013243)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()

</div>

---

## Why pyfai-zero-shot?

Standard 2-D XRD calibration tools (`pyFAI-calib2`, `DIOPTAS`, `DAWN`) give sub-pixel precision once initialized, but routinely get stuck in wrong-distance local minima when launched without an approximate distance, beam center, or manual ring picking—especially when switching between small-angle (`> 2 m`) and wide-angle setups or when the direct beam hits a detector corner.

`pyfai-zero-shot` solves geometry directly from a single image array, pixel size, and wavelength (or seeds `pyFAI` automatically into its `< 0.01%` optimum).

## Desktop Apps (11 Languages Live-Switch)

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — 5-stage visual flowchart workbench (`Data Import -> Auto Mask / Spline -> Zero-Prior Calibration -> Cake QA -> Export PONI & 2x2 Report`) supporting **11 languages** (`probes/i18n_wf.py`: 简体中文, English, 正體中文, 日本語, Français, Deutsch, Русский, Português, 한국어, Español, العربية with native RTL layout).
- **DiffBlindWizard (`probes/gui_wizard.py`)** — guided 5-step calibration wizard with color-coded confidence light.
- **Custom Calibrants (`probes/custom_phase.py`)** — load any `.cif` or `.D` phase file directly alongside the built-in standards (`LaB6` SRM 660c, `CeO2`, `Al2O3`, `Si`, `Quartz`, `AgBh`, `AgSt`).

## How It Works

1. **Adaptive Pre-processing:** `auto_offset` removes flat-panel negative pedestals (`< -5% p99.9`); `auto_mask` masks inter-module gaps, hot/bad pixels, and beamstop shadows.
2. **Multi-Scale Canvas Ladder (`R2`):** Starts on a fast `≤ 520 px` canvas and automatically escalates to `ceil(maxdim/2)` when scale compression merges close inner rings (`s ≥ 5`, `|ΔD|/D ≥ 10%`).
3. **Zero-Prior Centering & Phase Identification:** Combines Layer-0 photometric sharpness/Kåsa center checks, radial-symmetry + ray extrapolation seeds, cross-seed consensus voting, and full-profile Pearson correlation (`ps`).
4. **Full-Ring & Corner-Arc Refinement:** Native 2-D Taubin circle fitting on full rings; regularized `1440²` corner tilt + **2-parameter differential Pseudo-Voigt decoupling** (`r_i = D tan(2θ_i) + δr_0` with arc-normal center projection) on off-axis corner arcs (`az_cov < 50%`).

<div align="center">
<img src="assets/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and ablation" width="840">
</div>

## Benchmark Accuracy Across Synchrotron & Synthetic Data

| Test Suite | Measured Accuracy (Cold-Start, Zero Geometric Priors) |
| :--- | :--- |
| **DESY PETRA III P02.1** (13 scans, 230–2290 mm, Varex 2880² & PerkinElmer 2048², ~60 keV) | **13/13 (100%)** `LaB6`, **13/13 `< 0.12%`** distance error, median **`0.030%`**, R² `0.999999`, corner beam-center ΔC median **`0.74 px`** (`26.6 s/frame` CPU median) |
| **CLS 08B1-1 Real Minerals** (16 frames, 4 arsenic minerals from COD CIFs, mar3450) | **16/16 (100%)** convergence across `scorodite`, `realgar`, `orpiment`, `arsenopyrite`: **`246.90 ± 1.36 mm`**, MAD **`0.69 mm`**, `12.96 s` median |
| **ESRF D2AM `MX_ceria` & `Al2O3`** (150–450 mm; 52 frames incl. 1 dark frame) | `CeO2` phase-blind **7/7** (`0.139 mm` mean) / locked 2D **`0.1 mm`**; `Al2O3` **51/52** (blank dark frame rejected), `.spline` **`116.70 mm`** vs GT `116.86 mm` (`−0.16 mm`) |
| **Cross-Detector Transfer (`silx_zoo`)** | **5/5** phase hits: `Pilatus2M −0.007%`, `Eiger4M +0.041%`, `Eiger4M_v2 +0.311%`, `Eiger9M +0.008%`, `Pilatus1M` (`AgBh`, `+0.14%` in λ·D) |
| **100-Case Synthetic Boundary** (`Seed = 20261120`) | With **≥ 4 visible rings**: **41/41 (100%) `≤ 1.0%`**, **40/41 (97.6%) `≤ 0.20%`**, median **`0.0010%`** (`5–13` rings: `≤ 0.0019%`). With `≤ 3` rings, under-determined (`27.1%`) — automatically flagged by reliability tiers |
| **Standalone vs Seeding Standard `pyFAI`** (24 real frames, unweighted `cps.getList()`) | Zero-init `pyFAI` alone (calibrant given): **3/24 (12.5%)** (`0/8` corners, median `45.81%`) → `pyFAI-ZeroShot` (phase-blind): **24/24 (100%)** (`0.033%`) → **`pyFAI` seeded by `pyFAI-ZeroShot`**: **24/24 (100%)** at **`0.006%` / `0.11 px`** |

<details>
<summary><b>More Benchmark Visualizations (DESY Parity · Identifiability Limit · Multi-Facility · pyFAI Comparison)</b></summary>

<div align="center">
<img src="assets/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="assets/fig3_identifiability_bound.png" width="800"><br><br>
<img src="assets/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="assets/fig5_pyfai_comparison.png" width="820">
</div>

</details>

## Quick Start

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

## Citation

```bibtex
@software{pyfai_zero_shot_2026,
  author  = {Liu, Zi'ao and Liu, Jiaru and Liu, Haixia},
  title   = {pyfai-zero-shot: phase-blind, zero-prior geometry calibration of 2-D X-ray powder diffraction detectors},
  year    = {2026},
  url     = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot},
  license = {MIT}
}
```

## License & Acknowledgments

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for providing the NIST SRM 660c LaB6 flat-panel calibration datasets used in our benchmark validation, as well as the pyFAI community, ESRF D2AM (BM02), and CLS 08B1-1 for open diffraction datasets.

<sub>MIT License ([LICENSE](LICENSE)). Independent open-source tool compatible with pyFAI.</sub>
