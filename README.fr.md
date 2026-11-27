<div align="center">

# pyfai-zero-shot

**Calibration géométrique aveugle et sans a priori des détecteurs 2D de diffraction sur poudre<br>Anneaux complets et arcs partiels hors axe · Export pyFAI .poni**

🚫 Aucun étalon requis &nbsp;·&nbsp; 🚫 Aucune géométrie initiale &nbsp;·&nbsp; 🌍 Interface 11 langues (dont arabe RTL)<br>✅ Export <code>.poni</code> pyFAI &nbsp;·&nbsp; ✅ Diagnostic Cake &nbsp;·&nbsp; ✅ Import CIF personnalisé

[English](README.md) · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · **Français** · [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/built%20for-pyFAI-013243)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()

</div>

---

## Pourquoi pyfai-zero-shot ?

Les outils standards (`pyFAI-calib2`, `DIOPTAS`, `DAWN`) offrent une précision sub-pixel lorsqu'ils sont bien initialisés, mais échouent sans estimation de distance ou lorsque le faisceau frappe un coin du détecteur.

`pyfai-zero-shot` résout la phase et la géométrie à partir d'une seule image, de la taille de pixel et de la longueur d'onde, et exporte un `.poni` prêt pour `pyFAI`.

## Interface graphique (11 langues)

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — interface visuelle en 5 étapes avec changement de langue à la volée parmi **11 langues** (`probes/i18n_wf.py`: zh, en, zh_TW, ja, fr, de, ru, pt, ko, es, ar avec RTL).
- **DiffBlindWizard (`probes/gui_wizard.py`)** — assistant guidé en 5 étapes.
- **Phases personnalisées (`probes/custom_phase.py`)** — import direct de fichiers `.cif` ou `.D`.

## Fonctionnement

1. **Prétraitement :** `auto_offset` et `auto_mask` suppriment les piédestaux négatifs, espaces inter-modules, pixels chauds et ombres du beamstop.
2. **Échelle multi-résolution (`R2`) :** Démarre à `≤ 520 px` et passe à `ceil(maxdim/2)` en cas de divergence.
3. **Centrage & phase :** Garde netteté/Kåsa, symétrie radiale + rayons, consensus et score de profil `ps`.
4. **Raffinement :** Taubin 2D sur anneaux complets ; inclinaison `1440²` + **découplage Pseudo-Voigt différentiel à 2 paramètres** sur arcs partiels.

<div align="center">
<img src="assets/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and ablation" width="840">
</div>

## Performances mesurées (démarrage à froid)

| Jeu de test | Précision mesurée |
| :--- | :--- |
| **DESY PETRA III P02.1** (13 scans, 230–2290 mm) | **13/13 (100%)** `LaB6`, **13/13 `< 0.12%`**, médiane **`0.030%`**, R² `0.999999`, ΔC coin **`0.74 px`** |
| **CLS 08B1-1 Minéraux** (16 images, 4 minéraux) | **16/16 (100%)** convergence, **`246.90 ± 1.36 mm`**, MAD **`0.69 mm`** |
| **ESRF D2AM `MX_ceria` & `Al2O3`** | `CeO2` aveugle **7/7** (`0.139 mm`) ; `Al2O3` **51/52**, `.spline` **`116.70 mm`** vs GT `116.86 mm` |
| **Transfert `silx_zoo`** | **5/5** phases (`Pilatus2M −0.007%`, `Eiger4M +0.041%`, `Eiger9M +0.008%`) |
| **100 cas synthétiques** | Pour **≥ 4 anneaux** : **41/41 `≤ 1.0%`**, **40/41 `≤ 0.20%`**, médiane **`0.0010%`** (≤ 3 : `27.1%`) |
| **Comparaison `pyFAI`** (24 images) | `pyFAI` sans init **3/24 (12.5%)** → `pyFAI-ZeroShot` **24/24 (100%)** (`0.033%`) → `pyFAI` initialisé **24/24 (`0.006%` / `0.11 px`)** |

<details>
<summary><b>Afficher plus de graphiques de validation</b></summary>

<div align="center">
<img src="assets/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="assets/fig3_identifiability_bound.png" width="800"><br><br>
<img src="assets/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="assets/fig5_pyfai_comparison.png" width="820">
</div>

</details>

## Démarrage rapide

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

## Licence et remerciements

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for providing the NIST SRM 660c LaB6 flat-panel calibration datasets used in our benchmark validation, as well as the pyFAI community, ESRF D2AM (BM02), and CLS 08B1-1 for open diffraction datasets.

<sub>Licence MIT ([LICENSE](LICENSE)).</sub>
