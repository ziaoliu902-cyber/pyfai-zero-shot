<div align="center">

# pyfai-zero-shot

**Phase-blind, zero-prior geometric calibration of 2D powder X-ray diffraction detectors<br>across full rings and off-axis partial arcs**

🚫 No calibrant identity &nbsp;·&nbsp; 🚫 No initial geometry &nbsp;·&nbsp; 🌍 11-language GUI (incl. RTL Arabic)<br>✅ Standard pyFAI <code>.poni</code> out &nbsp;·&nbsp; ✅ Quantitative identifiability tiers &nbsp;·&nbsp; ✅ Frozen benchmark commit <code>f608e67</code>

**English** · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Français](README.fr.md) · [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Commit](https://img.shields.io/badge/frozen-f608e67-0072B2)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()
[![Manuscript](https://img.shields.io/badge/manuscript-Science_Advances_Research_Resource-D55E00)](manuscript/pyFAI-ZeroShot_SciAdv_ResearchResource.pdf)

</div>

---



## Desktop GUI (11 Languages)

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — 5-stage visual flowchart workbench with automatic bad-pixel/beamstop masking, Cake QA plots, and live language switching across **11 languages** (`probes/i18n_wf.py`: 简体中文, English, 正體中文, 日本語, Français, Deutsch, Русский, Português, 한국어, Español, العربية with native RTL layout).
- **DiffBlindWizard (`probes/gui_wizard.py`)** — guided step-by-step wizard for routine `.poni` generation.

## How It Works

1. **Adaptive Pre-processing:** `auto_offset` removes flat-panel negative pedestals (`< -5% p99.9`); `auto_mask` masks inter-module gaps, hot/invalid pixels, and beamstop shadows.
2. **Multi-Scale Canvas Ladder (`R2`):** Starts on a `≤ 520 px` canvas and conditionally escalates to `ceil(maxdim/2)` on scale-risk divergence (`s ≥ 5`, `|ΔD|/D ≥ 10%`).
3. **Zero-Prior Centering & Phase Arbitration:** Layer-0 photometric sharpness/Kåsa guard, ring-starvation guard (`|ln(D_RS/D_geo)| > 0.46`), radial-symmetry + ray fallback seeds, consensus voting, and Pearson profile score (`ps`).
4. **Full-Ring & Partial-Arc Polish:** Native 2D Taubin circle fitting on full rings; regularized `1440²` corner tilt + **2-parameter differential Pseudo-Voigt decoupling** (`r_i = D tan(2θ_i) + δr_0` with arc-normal center projection) on `< 50%` off-axis partial arcs.

<div align="center">
<img src="manuscript/figures/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and DESY 4-stage ablation" width="840">
</div>

## Verified Benchmarks (Frozen Commit `f608e67`)

| Benchmark | Result (`f608e67`, cold-start) |
| :--- | :--- |
| **DESY PETRA III P02.1** (13 scans, 230–2290 mm, Varex 2880² & PerkinElmer 2048², ~60 keV) | **13/13 (100%)** `LaB6`, **13/13 `< 0.12%`**, med **`0.030%`** (Locked OOD `0.028%`), R² `0.999999`, RMSE `0.82 mm`, corner ΔC **`0.74 px`** (`9.7×` gain), Mode B `8/8` bitwise identical |
| **CLS 08B1-1 Real Minerals** (16 frames, 4 arsenic minerals, COD/pymatgen CIFs, mar3450) | **16/16 (100%)** direct convergence (0 wrong basins), **`246.90 ± 1.36 mm`**, MAD **`0.69 mm`**, `ps = +0.583..+0.879`, med `12.96 s` |
| **ESRF D2AM `MX_ceria` & `Al2O3`** (7 distances 150–450 mm; 52 frames incl. 1 dark) | `CeO2` pure blind **7/7** (`0.139 mm` mean) / locked `phase_fit` **`0.1 mm`** mean & worst; `Al2O3` **51/52** (dark rejected), `.spline` mean **`116.70 mm`** vs GT `116.86 mm` (`−0.16 mm`) |
| **Cross-Detector `silx_zoo`** (known λ, `phases=None`) | **5/5** phase hit: `Pilatus2M −0.007%`, `Eiger4M +0.041%`, `Eiger4M_v2 +0.311%`, `Eiger9M +0.008%`, `Pilatus1M` (`AgBh`, `+0.14%` in λ·D) |
| **100-Case Synthetic Identifiability Bound** (`Seed = 20261120`) | Sharp boundary at `n_vis ≥ 4`: **41/41 (100%) `≤ 1.0%`**, **40/41 (97.6%) `≤ 0.20%`**, med **`0.0010%`** (`nvis = 5–13`: `31/31 ≤ 0.0019%`) vs `n_vis ≤ 3` (`27.1%`) |
| **Head-to-Head vs Unweighted `pyFAI`** (24 real frames) | Zero-init `pyFAI` (`P0`, calibrant given): **3/24 (12.5%)** (`0/8` DESY corners, med `45.81%`) → `Ours` (phase-blind): **24/24 (100%)** (`0.033%`) → Seeded `pyFAI` (`P1`): **24/24 (100%)** at **`0.006%` / `0.11 px`** |

<details>
<summary><b>More Publication Figures (DESY Blind Validation · Identifiability Bound · Multi-Facility · pyFAI Comparison)</b></summary>

<div align="center">
<img src="manuscript/figures/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="manuscript/figures/fig3_identifiability_bound.png" width="800"><br><br>
<img src="manuscript/figures/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="manuscript/figures/fig5_pyfai_comparison.png" width="820">
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

## Citation

```bibtex
@article{liu2026pyfaizeroshot,
  author  = {Liu, Zi'ao and Liu, Jiaru and Liu, Haixia},
  title   = {Phase-blind, zero-prior geometric calibration of two-dimensional powder X-ray diffraction detectors across full rings and off-axis partial arcs},
  year    = {2026},
  note    = {Science Advances Research Resource manuscript; frozen release commit f608e67},
  url     = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot}
}
```

## License & Acknowledgments

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for acquiring and providing the NIST SRM 660c LaB6 flat-panel calibration datasets.

<sub>MIT License ([LICENSE](LICENSE)). Multilingual READMEs are synchronized with frozen commit `f608e67`; the English version is authoritative.</sub>
