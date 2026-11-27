# Paper Numbers Freeze — `main @ f608e67` (2026-11-27)

> **Purpose:** Single authoritative source of verified experimental numbers for the *npj Computational Materials* manuscript and Figures 1–5. Every table below was executed cold-start (per-frame cache reset) on frozen commit **`f608e67`** (`D:\pzs-work`, SHA-256 synced with `D:\pyfai-project`). Do not cite legacy `2026-11-17` numbers (`v0.9.5`) without explicitly marking them as historical baselines.

---

## 0. Frozen Codebase & Core Regression

| Item | Verified Value |
| :--- | :--- |
| **Git Commit (`D:\pzs-work`)** | `main @ f608e67` (clean working tree; synced SHA-256 `9BF18936975B6A29104CECA0ABCABA2441B2B5FE81A5AAF0A7E9E5C70A5AD203` on `probes/solver_core.py`) |
| **Environment** | Windows 11 x64 CPU · Python 3.13 · `pyFAI 2025.12.1` |
| **Fast Core Regression** | `python -u D:\pyfai-project\probes\run_regression.py --fast` → **6 / 6 PASS (60.7 s)** (`py_compile` 13 files, `auto_mask`, `custom_phase`, `cake_report + Pilatus2M`, `batch_parallel + multiframe_calib`, `gui_solver` headless) |

---

## 1. Blind Synchrotron Validation: DESY PETRA III P02.1 (~60 keV, 13 Scans)

- **Protocol:** Read-only archive `D:\LaB6德国光源！！！` deduplicated to 13 independent scans (5 `center`, 8 `corner`; 12 on Varex XRD4343CT `2880² @ 150 µm`, 1 on PerkinElmer `2048² @ 200 µm`, $\lambda \approx 0.2072\text{–}0.2074\text{ Å}$). Pure blind Mode A (`phases=None, d_ref_mm=None`) clears all solver caches before each scan.
- **Aggregate Statistics (`f608e67`):**
  - **All 13 scans (`n=13`)**: `13/13 (100%)` `LaB6`, `13/13 < 0.12%` (`10/13 ≤ 0.05%`), `|ΔD|` median `0.438 mm`, mean `0.546 mm`, max `2.488 mm`; `|ΔD/D|` median **`0.0303%`**, mean `0.038%`, max **`0.1158%`**; `ΔC` median `0.80 px` (max `3.03 px`); linear regression **$R^2 = 0.999999$**, **$\text{RMSE} = 0.82\text{ mm}$**; total runtime **`437.8 s`**, median **`26.6 s/frame`**, mask median **`2.46 s`**.
  - **Locked OOD (`n=12`, excluding Dev `SDD230_center_MJ`)**: `12/12` `LaB6`, `|ΔD|` median `0.443 mm` (mean `0.583 mm`), `|ΔD/D|` median **`0.0281%`** (max `0.1158%`), `ΔC` median `1.01 px`, median `26.5 s`.
    - **Locked OOD `center` (`n=4`)**: `4/4` `LaB6`, `|ΔD|` med `0.238 mm`, `|ΔD/D|` med **`0.0148%`** (max `0.0347%`), `ΔC` med `1.46 px` (max `2.71 px`).
    - **Locked OOD `corner` (`n=8`)**: `8/8` `LaB6`, `|ΔD|` med `0.483 mm`, `|ΔD/D|` med **`0.0364%`** (max `0.1158%`), `ΔC` med **`0.74 px`** (max `3.03 px`).
  - **Mode B Nominal-SDD Check (`d_ref_mm` on all 8 `corner` scans)**: **`8/8 (100%)` bitwise identical** to pure blind Mode A (`harmonic_k = None`).

| # | Split | Scan ID | Geom | Detector | GT $D$ (mm) | GT Tilt | Pred Phase | Pred $D$ (mm) | $\Delta D$ (mm / %) | $\Delta C$ (px) | $m$ / rms / ps | Tier (Az Cov) | Time (s) |
|---:|:---|:---|:---|:---|---:|---:|:---|---:|---:|---:|:---|:---|---:|
| 1 | Dev | `SDD230_center_MJ` | center | Varex 2880² | 229.650 | 0.101° | LaB6 | 229.763 | +0.113 (+0.049%) | 0.13 | 11 / 0.497 / 0.648 | HIGH (100.0%) | 45.3 |
| 2 | Locked | `SDD1000_center_VB (1)` | center | Varex 2880² | 999.097 | 0.246° | LaB6 | 999.135 | +0.038 (+0.004%) | 0.45 | 12 / 0.515 / 0.592 | HIGH (100.0%) | 95.6 |
| 3 | Locked | `SDD1590_center_MJ` | center | Varex 2880² | 1585.753 | 0.117° | LaB6 | 1585.739 | −0.015 (−0.001%) | 1.22 | 12 / 0.187 / 0.462 | HIGH (100.0%) | 35.6 |
| 4 | Locked | `SDD1700_center_MJ` | center | Varex 2880² | 1698.979 | 0.978° | LaB6 | 1699.417 | +0.438 (+0.026%) | 2.71 | 11 / 0.055 / 0.730 | HIGH (100.0%) | 23.3 |
| 5 | Locked | `SDD2100_center_MJ` | center | Varex 2880² | 2098.268 | 0.093° | LaB6 | 2097.540 | −0.728 (−0.035%) | 1.69 | 7 / 0.147 / 0.474 | HIGH (100.0%) | 26.6 |
| 6 | Locked | `SDD300_corner_MJ` | corner | Varex 2880² | 299.973 | 0.137° | LaB6 | 299.816 | −0.157 (−0.052%) | 0.80 | 8 / 0.736 / 0.673 | PARTIAL_ARC (19.4%) | 44.1 |
| 7 | Locked | `SDD350_corner_VB` | corner | Perkin 2048² | 371.082 | 0.200° | LaB6 | 370.836 | −0.246 (−0.066%) | 2.17 | 9 / 0.332 / 0.520 | PARTIAL_ARC (30.6%) | 13.2 |
| 8 | Locked | `SDD1700_corner_MJ` | corner | Varex 2880² | 1722.589 | 0.239° | LaB6 | 1723.036 | +0.447 (+0.026%) | 0.64 | 12 / 0.243 / 0.490 | PARTIAL_ARC (30.6%) | 33.6 |
| 9 | Locked | `SDD1900_corner_MJ` | corner | Varex 2880² | 1898.685 | 0.195° | LaB6 | 1897.876 | −0.809 (−0.043%) | 1.30 | 12 / 0.201 / 0.471 | PARTIAL_ARC (30.6%) | 26.4 |
| 10 | Locked | `SDD2150_corner_VB` | corner | Varex 2880² | 2148.698 | 0.299° | LaB6 | 2146.210 | −2.488 (−0.116%) | 0.68 | 12 / 0.241 / 0.472 | PARTIAL_ARC (30.6%) | 23.2 |
| 11 | Locked | `SDD2190_corner_VB` | corner | Varex 2880² | 2187.695 | 0.188° | LaB6 | 2187.282 | −0.414 (−0.019%) | 0.51 | 12 / 0.199 / 0.474 | PARTIAL_ARC (30.6%) | 21.9 |
| 12 | Locked | `SDD2200_corner_MJ` | corner | Varex 2880² | 2197.667 | 0.271° | LaB6 | 2197.149 | −0.518 (−0.024%) | 0.35 | 12 / 0.278 / 0.475 | PARTIAL_ARC (30.6%) | 22.2 |
| 13 | Locked | `SDD2290_corner_MJ` | corner | Varex 2880² | 2287.749 | 0.169° | LaB6 | 2288.442 | +0.693 (+0.030%) | 3.03 | 12 / 0.198 / 0.474 | PARTIAL_ARC (30.6%) | 26.9 |

---

## 2. DESY 4-Stage Architectural Ablation

| Stage | Configuration | Phase Hit | `< 1.0%` | `< 0.20%` | All 13 $\|\Delta D/D\|$ Med / Max | Corner (n=8) $\|\Delta D/D\|$ Med / Max | Corner $\Delta C$ Med | Total / Med Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Base: single 480² canvas (`FB_AREA_MAX=0`), `mask=None` | 10 / 13 | 10 / 13 | 10 / 13 | 0.0303% / 945.3% | 0.0364% / 82.45% | 0.99 px | 163.1 s / 11.9 s |
| **A1** | `184c1aa`: + R2 multi-scale ladder + auto-mask (0-intercept 1D) | **13 / 13** | 12 / 13 | 12 / 13 | 0.0347% / 1.6208% | 0.1467% / 1.6208% | 7.21 px | 515.4 s / 32.1 s |
| **A2** | `0d3e493`: + coarse accel/caches + regularized 1440² 2D tilt + 1-param PV | **13 / 13** | **13 / 13** | 12 / 13 | 0.0303% / 0.4210% | **0.0364%** / 0.4210% | 0.79 px | 438.7 s / 26.6 s |
| **A3** | **`f608e67` (Full)**: + 2-param differential PV ($r = D\tan 2\theta + \delta r_0$) | **13 / 13** | **13 / 13** | **13 / 13** | **0.0303% / 0.1158%** | **0.0364% / 0.1158%** | **0.74 px** | **437.8 s / 26.6 s** |

---

## 3. CLS 08B1-1 Real Minerals (`16 frames`, mar3450 `4096² @ 73.242 µm`, 18 keV)

- **Protocol:** `data/cif/mineral_phases.json` (generated from COD CIFs + *pymatgen*), phase supplied (`phases=[mineral]`), no geometric prior, cold-start.
- **Result under `f608e67`:** **`16 / 16 (100%)` direct convergence** (`ps = +0.583 .. +0.879`, all `HIGH`, 0 wrong basins), median runtime `12.96 s/frame`; combined mean `246.90 mm`, sample SD `1.36 mm` (`0.55%`), median `247.24 mm`, **MAD `0.69 mm`**, range `244.48–249.95 mm`.

| Mineral | $n$ | Mean $D$ (mm) | Sample SD (mm) | Median $D$ (mm) | MAD (mm) | Min–Max (mm) | $m$ range | $ps$ range |
|---|---:|---:|---:|---:|---:|:---|:---:|:---:|
| `scorodite` | 4 | 247.27 | 2.24 | 247.32 | 1.42 | 244.48–249.95 | 12–14 | +0.706 .. +0.857 |
| `realgar` | 4 | 246.70 | 1.37 | 247.36 | 0.07 | 244.64–247.43 | 10–12 | +0.655 .. +0.879 |
| `orpiment` | 4 | 247.62 | 0.63 | 247.35 | 0.11 | 247.20–248.56 | 9–12 | +0.583 .. +0.734 |
| `arsenopyrite` | 4 | 246.01 | 0.11 | 245.99 | 0.07 | 245.92–246.14 | 14–15 | +0.678 .. +0.805 |

---

## 4. 100-Case Synthetic Identifiability Bound (`Seed = 20261120`, `LaB6`, `2048² @ 150 µm`, $\lambda = 0.35\text{ Å}$)

| Stratified Subset | $n$ | `LaB6` Phase Hit | `err ≤ 1.0%` | `err ≤ 0.20%` | Median `err` | P90 `err` | Max `err` | Median $\Delta C$ (px) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`nvis ≥ 4` (Well-posed)** | **41** | **41 / 41 (100%)** | **41 / 41 (100%)** | **40 / 41 (97.56%)** | **0.0010%** | 0.0571% | 0.2317% | **0.61 px** |
| **`nvis ≤ 3` (Under-determined)** | **59** | 20 / 59 (33.9%) | 16 / 59 (27.12%) | 9 / 59 (15.25%) | 25.233% | 50.486% | 64.535% | 26.37 px |
| `nvis = 1` | 18 | 1 / 18 | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 36.065% | 60.799% | 64.535% | 66.64 px |
| `nvis = 2` | 28 | 7 / 28 | 4 / 28 (14.29%) | 4 / 28 (14.29%) | 25.042% | 48.599% | 58.278% | 29.91 px |
| `nvis = 3` | 13 | 12 / 13 | 12 / 13 (92.31%) | 5 / 13 (38.46%) | 0.4704% | 0.5820% | 45.932% | 1.06 px |
| `nvis = 4` | 10 | 10 / 10 | 10 / 10 (100%) | 9 / 10 (90.0%) | 0.0433% | 0.1707% | 0.2317% | 1.78 px |
| `nvis = 5..11` (grouped) | 31 | 31 / 31 (100%) | 31 / 31 (100%) | 31 / 31 (100%) | ≤ 0.0012% | ≤ 0.0017% | 0.0019% | ~0.60 px |

---

## 5. ESRF D2AM (`MX_ceria` & `Al2O3`) & Known-$\lambda$ `silx_zoo`

1. **`phase_fit.py ceria` (locked 8-line `CeO2` + 2D refinement + tilt, 7 distances, `112.6 s`):** `7/7 PASS`, **mean $|\Delta D| = 0.1\text{ mm}$**, **worst $|\Delta D| = 0.1\text{ mm}$**; `rot1Δ` mean `-2.1 mrad`, `rot2Δ` mean `-2.5 mrad`.
2. **Pure Phase-Blind `MX_ceria` (`phases=None`, cold-start, 7 frames):** **`7/7 (100%)` `CeO2`**, `mean |ΔD| = 0.139 mm`, **`med |ΔD| = 0.111 mm`**, `max |ΔD| = 0.376 mm`, `med ΔC = 6.08 px`, `med 26.9 s`.
3. **`phase_fit.py alumina` (52 frames incl. 1 dark frame, `289.9 s`):** `51/52` solved (`dark_0001.edf` rejected), static med **`116.2 mm`** ($\Delta\text{GT} = -0.7\text{ mm}$, **`MAD = 0.68 mm`**); 51-frame mean + `.spline` joint: **`D = 116.70 mm`** (GT `116.86 mm`, $\Delta = -0.16\text{ mm}$), `rot = (+18.62, +10.82) mrad`.
4. **Known-$\lambda$ `silx_zoo` (`phases=None`, cold-start):** `Pilatus2M` `CeO2` `−0.007%`, `Eiger4M` `Al2O3` `+0.041%`, `Eiger4M_v2` `Al2O3` `+0.311%`, `Eiger9M` `CeO2` `+0.008%`, `Pilatus1M` `AgBh` `1638.037 mm` @ placeholder `1.0 Å`.

---

## 6. Head-to-Head Comparison with Standard `pyFAI` (`24 Real Frames`)

- **Setup:** Two rounds of `pyFAI` `RingExtraction(max_rings=15) + GeometryRefinement.refine3(fix=["wavelength"])` using standard unweighted control points (`cps.getList()`).

| Dataset | $n$ | `Ours` `<1%` / `<0.2%` | `Ours` Med $\|\Delta D/D\|$ / $\Delta C$ | `P0` `<1%` | `P0` Med $\|\Delta D/D\|$ | `P1` `<1%` / `<0.2%` | `P1` Med $\|\Delta D/D\|$ / $\Delta C$ |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DESY P02.1** (`LaB6`) | 13 | **13/13** / **13/13** | 0.030% / 0.80 px | 2/13 (0/8 corner) | 54.48% | **13/13** / **13/13** | 0.005% / **0.06 px** |
| **ESRF `MX_ceria`** (`CeO2`) | 7 | **7/7** / **7/7** | 0.052% / 6.08 px | 1/7 | 24.82% | **7/7** / **7/7** | 0.007% / **0.10 px** |
| **`silx_zoo` (known-λ)** | 4 | **4/4** / 3/4 | **0.024%** / 5.46 px | 0/4 | 38.14% | **4/4** / 2/4 | 0.403% / 2.18 px |
| **All Combined** | **24** | **24/24 (100%)** / **23/24** | **0.033%** / 2.44 px | **3/24 (12.5%)** | **45.81%** | **24/24 (100%)** / 22/24 | **0.006%** / **0.11 px** |
