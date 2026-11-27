<div align="center">

# pyfai-zero-shot

**既知相・初期ジオメトリ不要の2次元粉末X線回折検出器自動キャリブレーションエンジン<br>完全リングおよびオフアクシス部分弧対応・11言語GUI**

🚫 標準試料の指定不要 &nbsp;·&nbsp; 🚫 初期距離・ビーム中心不要 &nbsp;·&nbsp; 🌍 11言語デスクトップGUI（アラビア語RTL対応）<br>✅ pyFAI標準 <code>.poni</code> 出力 &nbsp;·&nbsp; ✅ 定量的な識別性ランク &nbsp;·&nbsp; ✅ 凍結コミット <code>f608e67</code>

[English](README.md) · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Français](README.fr.md) · **日本語**

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Commit](https://img.shields.io/badge/frozen-f608e67-0072B2)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![GUI Languages](https://img.shields.io/badge/GUI_languages-11_(incl._RTL)-009E73)]()
[![Manuscript](https://img.shields.io/badge/manuscript-Science_Advances_Research_Resource-D55E00)](manuscript/pyFAI-ZeroShot_SciAdv_ResearchResource.pdf)

</div>

---

## 論文プレプリント（Science Advances — Research Resource 形式）

- 📄 **本文 PDF（10ページ）：** [pyFAI-ZeroShot_SciAdv_ResearchResource.pdf](manuscript/pyFAI-ZeroShot_SciAdv_ResearchResource.pdf)
- 📊 **補足資料 PDF（Tables S1–S5）：** [pyFAI-ZeroShot_SciAdv_Supplementary.pdf](manuscript/pyFAI-ZeroShot_SciAdv_Supplementary.pdf)
- 🔒 **凍結ベンチマーク数値（`f608e67`）：** [docs/paper_numbers_freeze_2026-11-27_f608e67.md](docs/paper_numbers_freeze_2026-11-27_f608e67.md)

## 11言語デスクトップGUI

- **PyFAI-Workflow (`probes/gui_workflow.py`)** — 自動マスク・Cake品質診断を備えた5段階フローチャートUI。**`probes/i18n_wf.py` により11言語を即時切替可能**（zh, en, zh_TW, ja, fr, de, ru, pt, ko, es, ar RTL対応）。
- **DiffBlindWizard (`probes/gui_wizard.py`)** — 5ステップの対話型ウィザード。

## アルゴリズム概要

1. **適応的前処理：** `auto_offset` でフラットパネル負ペデスタルを除去、`auto_mask` でギャップ・不良画素・ビームストップ影を自動マスク。
2. **マルチスケール・キャンバス (`R2`)：** `≤ 520 px` から開始し、分岐検出時（`s ≥ 5`, `|ΔD|/D ≥ 10%`）に `ceil(maxdim/2)` へ自動昇格。
3. **ゼロプライア中心推定と相同定：** シャープネス/Kåsaガード、放射対称+レイ外挿シード、コンセンサス投票およびPearsonプロファイル相関 `ps`。
4. **精密化：** 完全リングはネイティブ2D Taubin円フィット、コーナー部分弧（`az_cov < 50%`）は正則化 `1440²` チルト＋**2パラメータ差分Pseudo-Voigt分離**（`r_i = D tan(2θ_i) + δr_0`）。

<div align="center">
<img src="manuscript/figures/fig1_pipeline_and_ablation.png" alt="Zero-prior pipeline and DESY 4-stage ablation" width="840">
</div>

## 検証ベンチマーク（`f608e67` コールドスタート）

| ベンチマーク | 結果（`f608e67`） |
| :--- | :--- |
| **DESY PETRA III P02.1**（13スキャン, 230–2290 mm） | **13/13 (100%)** `LaB6`, **13/13 `< 0.12%`**, 中央値 **`0.030%`** (OOD `0.028%`), R² `0.999999`, RMSE `0.82 mm`, コーナー中心 **`0.74 px`** (`9.7×`) |
| **CLS 08B1-1 鉱物**（16フレーム, ヒ素鉱物4種） | **16/16 (100%)** 直接収束, **`246.90 ± 1.36 mm`**, MAD **`0.69 mm`** |
| **ESRF D2AM `MX_ceria` & `Al2O3`** | `CeO2` ブラインド **7/7**（平均 `0.139 mm`）/ `phase_fit` **`0.1 mm`**; `Al2O3` **51/52**, `.spline` **`116.70 mm`** vs GT `116.86 mm` |
| **異検出器 `silx_zoo`** | **5/5** 相同定（`Pilatus2M −0.007%`, `Eiger4M +0.041%`, `Eiger9M +0.008%`） |
| **100例合成識別限界**（`Seed = 20261120`） | `n_vis ≥ 4` で急峻な遷移: **41/41 `≤ 1.0%`**, **40/41 `≤ 0.20%`**, 中央値 **`0.0010%`** vs `n_vis ≤ 3` (`27.1%`) |
| **標準 `pyFAI` 比較**（24実フレーム） | 初期値なし `pyFAI` **3/24 (12.5%)** → 本手法 **24/24 (100%)** (`0.033%`) → 本手法初期値 `pyFAI` **24/24 (100%)** (**`0.006%` / `0.11 px`**) |

<details>
<summary><b>その他の論文図</b></summary>

<div align="center">
<img src="manuscript/figures/fig2_desy_blind_validation.png" width="800"><br><br>
<img src="manuscript/figures/fig3_identifiability_bound.png" width="800"><br><br>
<img src="manuscript/figures/fig4_multifacility_benchmarks.png" width="820"><br><br>
<img src="manuscript/figures/fig5_pyfai_comparison.png" width="820">
</div>

</details>

## クイックスタート

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

## 引用

```bibtex
@article{liu2026pyfaizeroshot,
  author  = {Liu, Zi'ao and Liu, Jiaru and Liu, Haixia},
  title   = {Phase-blind, zero-prior geometric calibration of two-dimensional powder X-ray diffraction detectors across full rings and off-axis partial arcs},
  year    = {2026},
  note    = {Science Advances Research Resource manuscript; frozen release commit f608e67},
  url     = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot}
}
```

## ライセンスと謝辞

We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association HGF, for the provision of experimental facilities. Parts of this research were carried out at PETRA III beamline P02.1. Beamtime was allocated by an In-House contingent. We thank Dr. Martin Etter and the PETRA III P02.1 beamline team for acquiring and providing the NIST SRM 660c LaB6 flat-panel calibration datasets.

<sub>MITライセンス（[LICENSE](LICENSE)）。`f608e67` に同期済み。</sub>
