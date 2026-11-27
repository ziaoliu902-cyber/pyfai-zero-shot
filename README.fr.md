<div align="center">

# pyfai-zero-shot

**Calibration géométrique aveugle et sans a priori des détecteurs 2D de diffraction sur poudre<br>pour anneaux complets et arcs partiels hors axe**

🚫 Aucun étalon spécifié &nbsp;·&nbsp; 🚫 Aucune géométrie initiale &nbsp;·&nbsp; 🌍 Interface 11 langues (dont arabe RTL)<br>✅ Export <code>.poni</code> pyFAI &nbsp;·&nbsp; ✅ Niveaux d'identifiabilité &nbsp;·&nbsp; ✅ Commit gelé <code>f608e67</code>

[English](README.md) · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · **Français** · [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Commit](https://img.shields.io/badge/frozen-f608e67-0072B2)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()
[![Manuscript](https://img.shields.io/badge/manuscript-Science_Advances_Research_Resource-D55E00)](manuscript/pyFAI-ZeroShot_SciAdv_ResearchResource.pdf)

</div>

---

## Manuscrit & Prépublication (Science Advances — Research Resource)

- 📄 **Manuscrit principal PDF (10 p.) :** [pyFAI-ZeroShot_SciAdv_ResearchResource.pdf](manuscript/pyFAI-ZeroShot_SciAdv_ResearchResource.pdf)
- 📊 **Matériel supplémentaire PDF (Tableaux S1–S5) :** [pyFAI-ZeroShot_SciAdv_Supplementary.pdf](manuscript/pyFAI-ZeroShot_SciAdv_Supplementary.pdf)
- 🔒 **Chiffres gelés (`f608e67`) :** [docs/paper_numbers_freeze_2026-11-27_f608e67.md](docs/paper_numbers_freeze_2026-11-27_f608e67.md)

## Interface graphique (11 langues)

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — banc visuel en 5 étapes avec masquage automatique, diagnostics Cake et changement de langue à la volée parmi **11 langues** (`probes/i18n_wf.py`: zh, en, zh_TW, ja, fr, de, ru, pt, ko, es, ar avec RTL).
- **DiffBlindWizard (`probes/gui_wizard.py`)** — assistant guidé en 5 étapes.

## Fonctionnement

1. **Prétraitement adaptatif :** `auto_offset` supprime les piédestaux négatifs ; `auto_mask` masque les espaces inter-modules, pixels défectueux et ombres du beamstop.
2. **Échelle multi-résolution (`R2`) :** Démarre à `≤ 520 px` et passe à `ceil(maxdim/2)` en cas de divergence (`s ≥ 5`, `|ΔD|/D ≥ 10%`).
3. **Centrage & arbitrage de phase :** Garde netteté/Kåsa, symétrie radiale + rayons extrapolés, vote par consensus et corrélation de profil `ps`.
4. **Raffinement :** Ajustement circulaire 2D Taubin sur anneaux complets ; inclinaison régularisée `1440²` + **découplage Pseudo-Voigt différentiel à 2 paramètres** (`r_i = D tan(2θ_i) + δr_0`) sur arcs partiels (`az_cov < 50%`).

<div align="center">
<img src="manuscript/figures/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and DESY 4-stage ablation" width="840">
</div>

## Benchmarks vérifiés (`f608e67`, démarrage à froid)

| Benchmark | Résultat (`f608e67`) |
| :--- | :--- |
| **DESY PETRA III P02.1** (13 scans, 230–2290 mm) | **13/13 (100%)** `LaB6`, **13/13 `< 0.12%`**, médiane **`0.030%`** (OOD `0.028%`), R² `0.999999`, RMSE `0.82 mm`, ΔC coin **`0.74 px`** (`9.7×`) |
| **CLS 08B1-1 Minéraux** (16 images, 4 minéraux d'arsenic) | **16/16 (100%)** convergence directe, **`246.90 ± 1.36 mm`**, MAD **`0.69 mm`** |
| **ESRF D2AM `MX_ceria` & `Al2O3`** | `CeO2` aveugle **7/7** (`0.139 mm` moy.) / `phase_fit` **`0.1 mm`** ; `Al2O3` **51/52**, `.spline` **`116.70 mm`** vs GT `116.86 mm` |
| **Transfert `silx_zoo`** | **5/5** phases identifiées (`Pilatus2M −0.007%`, `Eiger4M +0.041%`, `Eiger9M +0.008%`) |
| **100 cas synthétiques** (`Seed = 20261120`) | Transition nette à `n_vis ≥ 4` : **41/41 `≤ 1.0%`**, **40/41 `≤ 0.20%`**, médiane **`0.0010%`** vs `n_vis ≤ 3` (`27.1%`) |
| **Comparaison avec `pyFAI`** (24 images réelles) | `pyFAI` sans init **3/24 (12.5%)** → `Ours` **24/24 (100%)** (`0.033%`) → `pyFAI` initialisé **24/24 (100%)** (**`0.006%` / `0.11 px`**) |

<details>
<summary><b>Autres figures de l'article</b></summary>

<div align="center">
<img src="manuscript/figures/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="manuscript/figures/fig3_identifiability_bound.png" width="800"><br><br>
<img src="manuscript/figures/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="manuscript/figures/fig5_pyfai_comparison.png" width="820">
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

## Licence et remerciements

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for acquiring and providing the NIST SRM 660c LaB6 flat-panel calibration datasets.

<sub>Licence MIT ([LICENSE](LICENSE)). Synchronisé avec le commit `f608e67`.</sub>
