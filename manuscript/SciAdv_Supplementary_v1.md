# Supplementary Materials — *Science Advances* Research Resource

**Title:** Phase-blind, zero-prior geometric calibration of two-dimensional powder X-ray diffraction detectors across full rings and off-axis partial arcs  
**Authors:** Zi'ao Liu, Jiaru Liu, and Haixia Liu  
**Frozen Code Commit:** `main @ f608e67`  

---

## Supplementary Table S1 | Complete per-scan DESY PETRA III P02.1 blind calibration results (Mode A vs Mode B, commit `f608e67`)

| # | Split | Scan ID | Geom | Detector (`shape`, pixel) | $\lambda$ (Å) | GT $D$ (mm) | GT Tilt (°) | Mode A $D$ (mm) | $\Delta D$ (mm / %) | $\Delta C$ (px) | $m$ / rms / $ps$ | Tier (`az_cov`) | Time (s) | Mode B `d_ref_mm` |
|---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|:---|:---|---:|:---:|
| 1 | Dev | `SDD230_center_MJ` | center | Varex 2880^2 (150 µm) | 0.207355 | 229.650 | 0.101 | 229.763 | +0.113 (+0.0494%) | 0.13 | 11 / 0.497 / 0.648 | HIGH (100.0%) | 45.29 | — (center) |
| 2 | Locked | `SDD1000_center_VB (1)` | center | Varex 2880^2 (150 µm) | 0.207333 | 999.097 | 0.246 | 999.135 | +0.038 (+0.0038%) | 0.45 | 12 / 0.515 / 0.592 | HIGH (100.0%) | 95.55 | — (center) |
| 3 | Locked | `SDD1590_center_MJ` | center | Varex 2880^2 (150 µm) | 0.207334 | 1585.753 | 0.117 | 1585.739 | -0.015 (-0.0009%) | 1.22 | 12 / 0.187 / 0.462 | HIGH (100.0%) | 35.56 | — (center) |
| 4 | Locked | `SDD1700_center_MJ` | center | Varex 2880^2 (150 µm) | 0.207353 | 1698.979 | 0.978 | 1699.417 | +0.438 (+0.0258%) | 2.71 | 11 / 0.055 / 0.730 | HIGH (100.0%) | 23.30 | — (center) |
| 5 | Locked | `SDD2100_center_MJ` | center | Varex 2880^2 (150 µm) | 0.207349 | 2098.268 | 0.093 | 2097.540 | -0.728 (-0.0347%) | 1.69 | 7 / 0.147 / 0.474 | HIGH (100.0%) | 26.64 | — (center) |
| 6 | Locked | `SDD300_corner_MJ` | corner | Varex 2880^2 (150 µm) | 0.207348 | 299.973 | 0.137 | 299.816 | -0.157 (-0.0525%) | 0.80 | 8 / 0.736 / 0.673 | PARTIAL_ARC (19.4%) | 44.14 | 299.816 mm (identical) |
| 7 | Locked | `SDD350_corner_VB` | corner | Perkin 2048^2 (200 µm) | 0.207200 | 371.082 | 0.200 | 370.836 | -0.246 (-0.0664%) | 2.17 | 9 / 0.332 / 0.520 | PARTIAL_ARC (30.6%) | 13.16 | 370.836 mm (identical) |
| 8 | Locked | `SDD1700_corner_MJ` | corner | Varex 2880^2 (150 µm) | 0.207314 | 1722.589 | 0.239 | 1723.036 | +0.447 (+0.0259%) | 0.64 | 12 / 0.243 / 0.490 | PARTIAL_ARC (30.6%) | 33.64 | 1723.036 mm (identical) |
| 9 | Locked | `SDD1900_corner_MJ` | corner | Varex 2880^2 (150 µm) | 0.207353 | 1898.685 | 0.195 | 1897.876 | -0.809 (-0.0426%) | 1.30 | 12 / 0.201 / 0.471 | PARTIAL_ARC (30.6%) | 26.37 | 1897.876 mm (identical) |
| 10 | Locked | `SDD2150_corner_VB` | corner | Varex 2880^2 (150 µm) | 0.207333 | 2148.698 | 0.299 | 2146.210 | -2.488 (-0.1158%) | 0.68 | 12 / 0.241 / 0.472 | PARTIAL_ARC (30.6%) | 23.16 | 2146.210 mm (identical) |
| 11 | Locked | `SDD2190_corner_VB` | corner | Varex 2880^2 (150 µm) | 0.207400 | 2187.695 | 0.188 | 2187.282 | -0.414 (-0.0189%) | 0.51 | 12 / 0.199 / 0.474 | PARTIAL_ARC (30.6%) | 21.93 | 2187.282 mm (identical) |
| 12 | Locked | `SDD2200_corner_MJ` | corner | Varex 2880^2 (150 µm) | 0.207334 | 2197.667 | 0.271 | 2197.149 | -0.518 (-0.0236%) | 0.35 | 12 / 0.278 / 0.475 | PARTIAL_ARC (30.6%) | 22.17 | 2197.149 mm (identical) |
| 13 | Locked | `SDD2290_corner_MJ` | corner | Varex 2880^2 (150 µm) | 0.207349 | 2287.749 | 0.169 | 2288.442 | +0.693 (+0.0303%) | 3.03 | 12 / 0.198 / 0.474 | PARTIAL_ARC (30.6%) | 26.92 | 2288.442 mm (identical) |

*Note:* Mode A uses `phases=None, d_ref_mm=None` with `auto_mask` enabled and all solver caches cleared before every scan. Median auto-mask time is `2.46 s`. On all 8 `corner` scans, Mode B (`d_ref_mm` supplied) yields bitwise identical geometry to pure blind Mode A (`harmonic_k = None`).

---

## Supplementary Table S2 | Per-scan progression across the 4-stage DESY ablation (`A0 -> A1 -> A2 -> A3`)

| Scan ID | Geom | GT $D$ (mm) | A0 (`480^2`, no mask) | A1 (`184c1aa` R2+mask) | A2 (`0d3e493` Tilt+1p) | A3 (`f608e67` Full 2p) |
|:---|:---|---:|:---|:---|:---|:---|
| `SDD230_center_MJ` | center | 229.650 | AgBh +945.33% (4.4 px) | LaB6 +0.0494% (0.13 px) | LaB6 +0.0494% (0.13 px) | LaB6 +0.0494% (0.13 px) |
| `SDD1000_center_VB (1)` | center | 999.097 | AgBh +103.17% (137.8 px) | LaB6 +0.0038% (0.45 px) | LaB6 +0.0038% (0.45 px) | LaB6 +0.0038% (0.45 px) |
| `SDD1590_center_MJ` | center | 1585.753 | LaB6 +0.00% (0.1 px) | LaB6 +0.0006% (0.16 px) | LaB6 -0.0009% (1.22 px) | LaB6 -0.0009% (1.22 px) |
| `SDD1700_center_MJ` | center | 1698.979 | LaB6 +0.03% (2.7 px) | LaB6 +0.0258% (2.71 px) | LaB6 +0.0258% (2.71 px) | LaB6 +0.0258% (2.71 px) |
| `SDD2100_center_MJ` | center | 2098.268 | LaB6 -0.00% (1.1 px) | LaB6 -0.0347% (1.69 px) | LaB6 -0.0347% (1.69 px) | LaB6 -0.0347% (1.69 px) |
| `SDD300_corner_MJ` | corner | 299.973 | Quartz +82.45% (3.5 px) | LaB6 +1.6208% (4.08 px) | LaB6 +0.4210% (0.90 px) | LaB6 -0.0525% (0.80 px) |
| `SDD350_corner_VB` | corner | 371.082 | LaB6 -0.07% (2.2 px) | LaB6 -0.1448% (2.39 px) | LaB6 -0.1446% (2.39 px) | LaB6 -0.0664% (2.17 px) |
| `SDD1700_corner_MJ` | corner | 1722.589 | LaB6 +0.03% (0.6 px) | LaB6 -0.0052% (7.48 px) | LaB6 +0.0259% (0.64 px) | LaB6 +0.0259% (0.64 px) |
| `SDD1900_corner_MJ` | corner | 1898.685 | LaB6 -0.04% (1.3 px) | LaB6 -0.1874% (10.00 px) | LaB6 -0.0426% (1.30 px) | LaB6 -0.0426% (1.30 px) |
| `SDD2150_corner_VB` | corner | 2148.698 | LaB6 -0.12% (0.7 px) | LaB6 -0.1485% (9.15 px) | LaB6 -0.1158% (0.68 px) | LaB6 -0.1158% (0.68 px) |
| `SDD2190_corner_VB` | corner | 2187.695 | LaB6 -0.02% (0.5 px) | LaB6 +0.0298% (6.93 px) | LaB6 -0.0189% (0.51 px) | LaB6 -0.0189% (0.51 px) |
| `SDD2200_corner_MJ` | corner | 2197.667 | LaB6 -0.02% (0.3 px) | LaB6 -0.1583% (10.57 px) | LaB6 -0.0236% (0.35 px) | LaB6 -0.0236% (0.35 px) |
| `SDD2290_corner_MJ` | corner | 2287.749 | LaB6 +0.03% (3.0 px) | LaB6 +0.0098% (3.80 px) | LaB6 +0.0303% (3.03 px) | LaB6 +0.0303% (3.03 px) |

---

## Supplementary Table S3 | CLS 08B1-1 arsenic mineral benchmark (`mar3450`, `4096^2`, `73.242 µm`, `18 keV`, COD/pymatgen CIF cards)

| Mineral | Frame # | Recovered $D$ (mm) | Matched Rings $m$ | Radial RMS (px) | Pattern Score $ps$ | Matched Filter | Confidence Tier | Runtime (s) |
|:---|---:|---:|---:|---:|---:|---:|:---|---:|
| `scorodite` | 1 | 247.113 | 12 | 1.179 | +0.8571 | +0.5610 | HIGH | 14.06 |
| `scorodite` | 2 | 247.528 | 12 | 1.415 | +0.8450 | +0.6023 | HIGH | 14.72 |
| `scorodite` | 3 | 244.476 | 13 | 2.436 | +0.7292 | +0.5749 | HIGH | 14.39 |
| `scorodite` | 4 | 249.946 | 14 | 1.302 | +0.7061 | +0.6138 | HIGH | 13.72 |
| `realgar` | 1 | 247.431 | 11 | 1.103 | +0.8703 | +0.4892 | HIGH | 13.43 |
| `realgar` | 2 | 244.644 | 10 | 0.621 | +0.8143 | +0.5151 | HIGH | 13.36 |
| `realgar` | 3 | 247.417 | 12 | 1.038 | +0.8794 | +0.5093 | HIGH | 13.76 |
| `realgar` | 4 | 247.300 | 12 | 1.554 | +0.6553 | +0.5286 | HIGH | 12.92 |
| `orpiment` | 1 | 247.198 | 11 | 0.742 | +0.7337 | +0.4483 | HIGH | 12.99 |
| `orpiment` | 2 | 247.282 | 10 | 1.177 | +0.6229 | +0.4632 | HIGH | 12.77 |
| `orpiment` | 3 | 248.556 | 12 | 0.857 | +0.5835 | +0.4256 | HIGH | 12.60 |
| `orpiment` | 4 | 247.425 | 9 | 1.228 | +0.6886 | +0.4396 | HIGH | 12.74 |
| `arsenopyrite` | 1 | 246.138 | 14 | 2.016 | +0.8053 | +0.6089 | HIGH | 12.75 |
| `arsenopyrite` | 2 | 245.922 | 15 | 0.316 | +0.6900 | +0.6091 | HIGH | 12.91 |
| `arsenopyrite` | 3 | 246.063 | 15 | 0.280 | +0.6781 | +0.6057 | HIGH | 12.77 |
| `arsenopyrite` | 4 | 245.921 | 15 | 0.378 | +0.6901 | +0.6092 | HIGH | 12.55 |

**Aggregate across all 16 frames (`0` rejected):** Mean = `246.90 mm`, Sample SD = `1.36 mm` (`0.55%`), Median = `247.24 mm`, MAD = `0.69 mm`, Min–Max = `244.48–249.95 mm`, Median Runtime = `12.96 s`.

---

## Supplementary Table S4 | ESRF D2AM `MX_ceria` (pure blind vs `phase_fit`) and `silx_zoo` cross-detector results (`f608e67`)

### Part A: ESRF D2AM `MX_ceria` (Pilatus 6M, `172 µm`, $\lambda = 0.972386\text{ Å}$)

| Nominal (mm) | GT $D$ (mm) | Blind Phase | Pure Blind $D$ (mm) | Blind $\Delta D$ (mm / %) | Blind $\Delta C$ (px) | $m$ / rms / $ps$ | `phase_fit` $\Delta D$ (mm) | Blind Runtime (s) |
|---:|---:|:---|---:|---:|---:|:---|---:|---:|
| 150 | 149.045 | `CeO2` | 148.968 | -0.077 (-0.0518%) | 6.21 | 9 / 0.358 / 0.755 | -0.1 | 35.01 |
| 200 | 199.173 | `CeO2` | 199.062 | -0.111 (-0.0556%) | 4.78 | 9 / 0.657 / 0.740 | -0.1 | 27.10 |
| 250 | 249.246 | `CeO2` | 249.211 | -0.036 (-0.0143%) | 4.84 | 8 / 0.873 / 0.745 | -0.0 | 28.34 |
| 300 | 299.322 | `CeO2` | 299.107 | -0.215 (-0.0717%) | 9.38 | 6 / 0.939 / 0.771 | -0.0 | 26.91 |
| 350 | 349.383 | `CeO2` | 349.007 | -0.376 (-0.1075%) | 6.08 | 5 / 0.058 / 0.816 | -0.1 | 24.63 |
| 400 | 399.459 | `CeO2` | 399.346 | -0.112 (-0.0281%) | 4.22 | 5 / 0.258 / 0.805 | -0.1 | 22.37 |
| 450 | 449.480 | `CeO2` | 449.526 | +0.046 (+0.0102%) | 6.84 | 4 / 0.058 / 0.813 | -0.0 | 25.57 |

### Part B: Known-$\lambda$ `silx_zoo` Cross-Detector Benchmark (`phases=None`)

| Dataset | Shape | Pixel (µm) | $\lambda$ (Å) | Exp / Pred Phase | GT $D$ (mm) | Pred $D$ (mm) | $\Delta D$ (mm / %) | $\Delta C$ (px) | $m$ / rms / $ps$ | Time (s) |
|:---|:---|---:|---:|:---|---:|---:|---:|---:|:---|---:|
| `Pilatus2M` | 1679x1475 | 172 | 0.171961 | `CeO2` / `CeO2` | 799.942 | 799.888 | -0.054 (-0.0068%) | 0.69 | 8 / 0.589 / 0.458 | 8.89 |
| `Eiger4M` | 2167x2070 | 75 | 0.921816 | `Al2O3` / `Al2O3` | 162.558 | 162.625 | +0.066 (+0.0408%) | 12.31 | 7 / 0.188 / 0.448 | 37.97 |
| `Eiger4M_v2` | 2167x2070 | 75 | 0.924973 | `Al2O3` / `Al2O3` | 161.933 | 162.437 | +0.504 (+0.3110%) | 4.89 | 7 / 0.142 / 0.448 | 37.56 |
| `Eiger9M` | 3262x3108 | 75 | 0.373800 | `CeO2` / `CeO2` | 213.203 | 213.220 | +0.017 (+0.0078%) | 6.02 | 8 / 0.534 / 0.474 | 73.98 |
| `Pilatus1M` | 1043x981 | 172 | 1.000000 | `AgBh` / `AgBh` | 1583.231 | 1638.037 | +54.806 (+3.4617%) | 0.82 | 8 / 0.709 / 0.924 | 16.66 |

*Note on `Pilatus1M`:* The public `.poni` file uses a nominal placeholder wavelength of `1.000000 Å` (`GT = 1583.231 mm`). Relative to the actual ~12 keV beamline energy, the recovered physical product $\lambda \cdot D$ matches within `+0.14%`.

---

## Supplementary Table S5 | Per-frame comparison with standard unweighted `pyFAI` across 24 real frames (`P0` vs `Ours` vs `P1`)

| Dataset | Scan / Frame | Calibrant | GT $D$ (mm) | `Ours` $D$ (mm) (`dD%` / `dC`) | `P0` Zero-Init `pyFAI` (`dD%` / `dC`) | `P1` Seeded `pyFAI` (`dD%` / `dC` / `tilt`) |
|:---|:---|:---|---:|:---|:---|:---|
| DESY P02.1 | `SDD230_center_MJ` | `LaB6` | 229.650 | 229.763 (`+0.0494%` / `0.13 px`) | 100.093 (`-56.42%` / `10.79 px`) | 229.765 (`+0.0503%` / `0.04 px` / `0.152°`) |
| DESY P02.1 | `SDD1000_center_VB (1)` | `LaB6` | 999.097 | 999.135 (`+0.0038%` / `0.45 px`) | 1000.793 (`+0.17%` / `78.69 px`) | 999.091 (`-0.0006%` / `0.02 px` / `0.241°`) |
| DESY P02.1 | `SDD1590_center_MJ` | `LaB6` | 1585.753 | 1585.739 (`-0.0009%` / `1.22 px`) | 1950.213 (`+22.98%` / `47.41 px`) | 1585.808 (`+0.0035%` / `0.08 px` / `0.120°`) |
| DESY P02.1 | `SDD1700_center_MJ` | `LaB6` | 1698.979 | 1699.417 (`+0.0258%` / `2.71 px`) | 1972.080 (`+16.07%` / `46.67 px`) | 1699.478 (`+0.0294%` / `2.73 px` / `0.122°`) |
| DESY P02.1 | `SDD2100_center_MJ` | `LaB6` | 2098.268 | 2097.540 (`-0.0347%` / `1.69 px`) | 2087.144 (`-0.53%` / `4.09 px`) | 2098.277 (`+0.0004%` / `0.05 px` / `0.095°`) |
| DESY P02.1 | `SDD300_corner_MJ` | `LaB6` | 299.973 | 299.816 (`-0.0525%` / `0.80 px`) | 1842.778 (`+514.31%` / `2008.45 px`) | 300.092 (`+0.0396%` / `0.02 px` / `0.265°`) |
| DESY P02.1 | `SDD350_corner_VB` | `LaB6` | 371.082 | 370.836 (`-0.0664%` / `2.17 px`) | 100.037 (`-73.04%` / `1410.63 px`) | 371.073 (`-0.0025%` / `0.02 px` / `0.139°`) |
| DESY P02.1 | `SDD1700_corner_MJ` | `LaB6` | 1722.589 | 1723.036 (`+0.0259%` / `0.64 px`) | 100.133 (`-94.19%` / `1920.83 px`) | 1722.499 (`-0.0052%` / `0.21 px` / `0.185°`) |
| DESY P02.1 | `SDD1900_corner_MJ` | `LaB6` | 1898.685 | 1897.876 (`-0.0426%` / `1.30 px`) | 99.526 (`-94.76%` / `1904.33 px`) | 1898.697 (`+0.0006%` / `0.04 px` / `0.203°`) |
| DESY P02.1 | `SDD2150_corner_VB` | `LaB6` | 2148.698 | 2146.210 (`-0.1158%` / `0.68 px`) | 1001.388 (`-53.40%` / `1905.47 px`) | 2148.930 (`+0.0108%` / `0.06 px` / `0.348°`) |
| DESY P02.1 | `SDD2190_corner_VB` | `LaB6` | 2187.695 | 2187.282 (`-0.0189%` / `0.51 px`) | 1000.341 (`-54.27%` / `1871.25 px`) | 2188.093 (`+0.0182%` / `0.15 px` / `0.253°`) |
| DESY P02.1 | `SDD2200_corner_MJ` | `LaB6` | 2197.667 | 2197.149 (`-0.0236%` / `0.35 px`) | 1000.272 (`-54.48%` / `1927.70 px`) | 2197.619 (`-0.0022%` / `0.12 px` / `0.271°`) |
| DESY P02.1 | `SDD2290_corner_MJ` | `LaB6` | 2287.749 | 2288.442 (`+0.0303%` / `3.03 px`) | 99.960 (`-95.63%` / `1895.89 px`) | 2288.088 (`+0.0148%` / `0.14 px` / `0.203°`) |
| ESRF MX_ceria | `ceria_150` | `CeO2` | 149.045 | 148.968 (`-0.0518%` / `6.21 px`) | 287.715 (`+93.04%` / `14.72 px`) | 149.045 (`+0.0000%` / `0.24 px` / `0.500°`) |
| ESRF MX_ceria | `ceria_200` | `CeO2` | 199.173 | 199.062 (`-0.0556%` / `4.78 px`) | 306.722 (`+54.00%` / `18.57 px`) | 199.177 (`+0.0022%` / `0.04 px` / `0.492°`) |
| ESRF MX_ceria | `ceria_250` | `CeO2` | 249.246 | 249.211 (`-0.0143%` / `4.84 px`) | 303.852 (`+21.91%` / `9.60 px`) | 249.258 (`+0.0048%` / `0.05 px` / `0.491°`) |
| ESRF MX_ceria | `ceria_300` | `CeO2` | 299.322 | 299.107 (`-0.0717%` / `9.38 px`) | 299.348 (`+0.01%` / `0.02 px`) | 299.342 (`+0.0067%` / `0.10 px` / `0.490°`) |
| ESRF MX_ceria | `ceria_350` | `CeO2` | 349.383 | 349.007 (`-0.1075%` / `6.08 px`) | 296.125 (`-15.24%` / `12.04 px`) | 349.411 (`+0.0080%` / `0.11 px` / `0.485°`) |
| ESRF MX_ceria | `ceria_400` | `CeO2` | 399.459 | 399.346 (`-0.0281%` / `4.22 px`) | 300.311 (`-24.82%` / `7.96 px`) | 399.485 (`+0.0066%` / `0.84 px` / `0.478°`) |
| ESRF MX_ceria | `ceria_450` | `CeO2` | 449.480 | 449.526 (`+0.0102%` / `6.84 px`) | 300.277 (`-33.19%` / `8.38 px`) | 449.517 (`+0.0082%` / `0.08 px` / `0.509°`) |
| silx_zoo | `Pilatus2M` | `CeO2` | 799.942 | 799.888 (`-0.0068%` / `0.69 px`) | 980.359 (`+22.55%` / `0.53 px`) | 799.963 (`+0.0026%` / `0.11 px` / `0.071°`) |
| silx_zoo | `Eiger4M` | `alpha_Al2O3` | 162.558 | 162.625 (`+0.0408%` / `12.31 px`) | 100.428 (`-38.22%` / `234.06 px`) | 163.917 (`+0.8361%` / `5.50 px` / `0.984°`) |
| silx_zoo | `Eiger4M_v2` | `alpha_Al2O3` | 161.933 | 162.437 (`+0.3110%` / `4.89 px`) | 100.292 (`-38.07%` / `234.92 px`) | 163.232 (`+0.8020%` / `4.15 px` / `0.864°`) |
| silx_zoo | `Eiger9M` | `CeO2` | 213.203 | 213.220 (`+0.0078%` / `6.02 px`) | 984.037 (`+361.55%` / `44.26 px`) | 213.195 (`-0.0037%` / `0.21 px` / `1.189°`) |

---

## Supplementary Note | `pyFAI 2025.12.1` RingExtraction and Unweighted Control-Point Protocol

All `pyFAI` baseline (`P0`) and seeded refinement (`P1`) runs used `pyFAI 2025.12.1` on Python 3.13 with two technical safeguards:
1. **`RingExtraction._calculate_num_of_points_to_keep` signature bug:** In `pyFAI 2025.12.1`, calling `extract_control_points()` with `RingExtraction` invokes `_calculate_num_of_points_to_keep` using an API mismatch on the radial/chi array wrapper. Both `P0` and `P1` applied the exact deterministic patch (`_npk_fixed`) computing unique valid degree sectors from `center_array(sh, unit=pyFAI.units.CHI_RAD, scale=False)`.
2. **Unweighted `cps.getList()` vs raw-intensity weighting `getWeightedList(img)`:** Standard `pyFAI-calib2` refines geometry using unweighted control points (`cps.getList()`). In preliminary runs using raw image intensity weights `cps.getWeightedList(img)` on high-dynamic-range detectors (DESY Varex `10^4–10^6` counts, Pilatus/Eiger counts), SciPy's `fmin_slsqp` solver encountered objective magnitudes near `10^7`, raising `Inequality constraints incompatible (Exit mode 4)` and discarding parameter updates (`dist` left frozen at the initial seed `D0`). Switching to standard unweighted `cps.getList()` resolved the numerical conditioning artifact cleanly (`Pilatus2M` $\chi^2$ dropped from `3.60e-3` to `1.05e-3`), yielding a fair comparison where `P0` converges on `3/24` frames while `P1` (seeded by our zero-prior `.poni`) converges on `24/24` frames (`0.006%` median distance error, `0.11 px` median center error).

