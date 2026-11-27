<div align="center">

# pyfai-zero-shot

**Phasenblinde, vorwissenfreie geometrische Kalibrierung von 2D-Röntgen-Pulverdetektoren<br>Für Vollringe und off-axis Teilbögen · Direkter pyFAI-PONI-Export**

🚫 Keine Kalibrant-Vorgabe &nbsp;·&nbsp; 🚫 Keine Startgeometrie &nbsp;·&nbsp; 🌍 11-Sprachen-GUI (inkl. RTL Arabisch)<br>✅ Standard-pyFAI-<code>.poni</code>-Export &nbsp;·&nbsp; ✅ Cake-Qualitätsprüfung &nbsp;·&nbsp; ✅ Eigene CIF-Dateien

[English](README.md) · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · **Deutsch** · [Русский](README.ru.md) · [Français](README.fr.md) · [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/built%20for-pyFAI-013243)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()

</div>

---

## Warum pyfai-zero-shot?

Etablierte Kalibrierungstools (`pyFAI-calib2`, `DIOPTAS`, `DAWN`) erreichen höchste Präzision bei guter Initialisierung, konvergieren jedoch ohne geschätzte Detektordistanz oder bei Strahlpositionen in der Detektorecke oft in falsche Minima.

`pyfai-zero-shot` bestimmt Phase und Geometrie automatisch allein aus Bildarray, Pixelgröße und Wellenlänge und übergibt saubere `.poni`-Startwerte direkt an `pyFAI`.

## Desktop-GUI (11 Sprachen)

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — 5-stufige Workflow-Oberfläche mit Auto-Masking, Cake-Diagnose und Live-Sprachumschaltung über **11 Sprachen** (`probes/i18n_wf.py`: zh, en, zh_TW, ja, fr, de, ru, pt, ko, es, ar inkl. RTL).
- **DiffBlindWizard (`probes/gui_wizard.py`)** — geführter 5-Schritte-Assistent.
- **Eigene Phasen (`probes/custom_phase.py`)** — beliebige `.cif`- oder `.D`-Dateien laden.

## Funktionsweise

1. **Vorverarbeitung:** `auto_offset` entfernt negative Flachdetektor-Offsets; `auto_mask` maskiert Modulspalten, Defektpixel und Beamstop-Schatten.
2. **Mehrskalige Canvas-Leiter (`R2`):** Start bei `≤ 520 px`, automatische Eskalation auf `ceil(maxdim/2)` bei Divergenz.
3. **Zentrierung & Phasenwahl:** Schärfe-/Kåsa-Guard, Radialsymmetrie + Strahl-Seeds, Konsenswahl und Profilkorrelation `ps`.
4. **Feinjustierung:** 2D-Taubin-Kreisfit auf Vollringen; regularisierte `1440²`-Ecken-Kippung + **2-Parameter-Differential-Pseudo-Voigt-Entkopplung** auf Teilbögen (`az_cov < 50%`).

<div align="center">
<img src="assets/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and ablation" width="840">
</div>

## Verifizierte Messgenauigkeit (Kaltstart)

| Datensatz | Gemessene Genauigkeit |
| :--- | :--- |
| **DESY PETRA III P02.1** (13 Scans, 230–2290 mm) | **13/13 (100%)** `LaB6`, **13/13 `< 0.12%`**, Median **`0.030%`**, R² `0.999999`, Ecken-ΔC **`0.74 px`** |
| **CLS 08B1-1 Minerale** (16 Frames, 4 Arsenminerale) | **16/16 (100%)** Konvergenz, **`246.90 ± 1.36 mm`**, MAD **`0.69 mm`** |
| **ESRF D2AM `MX_ceria` & `Al2O3`** | `CeO2` blind **7/7** (`0.139 mm` Mittel); `Al2O3` **51/52**, `.spline` **`116.70 mm`** vs. GT `116.86 mm` |
| **Detektor-Transfer `silx_zoo`** | **5/5** Phasentreffer (`Pilatus2M −0.007%`, `Eiger4M +0.041%`, `Eiger9M +0.008%`) |
| **100 synthetische Geometrien** | Ab `n_vis ≥ 4` Ringen: **41/41 `≤ 1.0%`**, **40/41 `≤ 0.20%`**, Median **`0.0010%`** (≤ 3 Ringe: `27.1%`) |
| **Vergleich mit `pyFAI`** (24 reale Frames) | `pyFAI` Zero-Init **3/24 (12.5%)** → `pyFAI-ZeroShot` **24/24 (100%)** (`0.033%`) → mit Seed an `pyFAI` **24/24 (`0.006%` / `0.11 px`)** |

<details>
<summary><b>Weitere Benchmark-Grafiken anzeigen</b></summary>

<div align="center">
<img src="assets/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="assets/fig3_identifiability_bound.png" width="800"><br><br>
<img src="assets/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="assets/fig5_pyfai_comparison.png" width="820">
</div>

</details>

## Schnellstart

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

## Zitation

```bibtex
@software{pyfai_zero_shot_2026,
  author  = {Liu, Zi'ao and Liu, Jiaru and Liu, Haixia},
  title   = {pyfai-zero-shot: phase-blind, zero-prior geometry calibration of 2-D X-ray powder diffraction detectors},
  year    = {2026},
  url     = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot},
  license = {MIT}
}
```

## Lizenz & Danksagung

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for providing the NIST SRM 660c LaB6 flat-panel calibration datasets used in our benchmark validation, as well as the pyFAI community, ESRF D2AM (BM02), and CLS 08B1-1 for open diffraction datasets.

<sub>MIT-Lizenz ([LICENSE](LICENSE)).</sub>
