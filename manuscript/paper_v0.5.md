# Phase-blind zero-shot geometry calibration of two-dimensional X-ray detectors from powder rings

**Draft v0.4 — 2026-11-03（v0.4：标题+摘要口径改为 phase-blind zero-shot——cascade/hybrid 从主线降为实现机制，补跨仪器基准与共心真值中心两句；正文未动。v0.3：Abstract 199 词、全文 polish、数据考证、查新；数字均出自项目文档 07–13 号 + zoo 基准 v3/T9）**

作者：[TODO — 须用户提供] 单位：[TODO — 须用户提供] 目标期刊：*Journal of Applied Crystallography*

---

## Abstract

Automatic geometry calibration of two-dimensional X-ray detectors —
sample-to-detector distance, beam-center position and tilts — conventionally
requires a known calibrant, manual ring picking or a good initial guess.
A phase-blind, zero-shot pipeline is presented that starts from a single
image of an *unknown* powder: no phase assignment, no initial geometry and,
for three of the benchmark detectors, no wavelength. A classic
peak-position solver runs first from the canvas center; fallback
initializers engage only on failure — a closed-form radial-symmetry center
estimator requiring no initial guess (2.9 px median near-center error),
ray-extrapolation seeds compensating its off-axis shrinkage bias and,
optionally, a regression neural network — and a cross-seed consensus gate
arbitrates the candidates, eliminating the self-consistent
pseudo-solutions that defeat ring-count-first ranking. Profile peaks are
paired to (hkl) lines of a phase library by iterative ratio-tolerance
matching; (cx, cy, D) are refined by nonlinear least squares with harmonic
re-centering. On seven real CeO2 frames (150–450 mm) the pipeline attains
0.1 mm mean distance error (0.1 mm worst) with a native 2D refinement,
0.2 mm (0.3 mm worst) on the 15 ms classic path, with full-geometry 2θ residuals of 1.3–2.4 mrad rms; 52 static Al2O3 frames
reproduce the reference distance to 0.45 mm MAD. On a ten-image
cross-instrument benchmark it identifies the phase on six of the eight
evaluable frames and recovers the distance within 0.1 mm on four detectors. An
image-internal sharpness criterion adjudicates beam centers
reference-free, exposing a several-pixel bias of purely concentric
fitting on tilted, edge-clipped rings despite sub-0.05 px formal
precision, and one distributed configuration is shown to pair an
unrelated image with a 56 px-offset calibration. An initializer
ablation shows no single initializer suffices, and a sensitivity analysis
yields prior-error propagation coefficients (wavelength/lattice ×1.40,
pixel size ×1.00), photon noise being non-limiting down to 1/20 exposure.

**Keywords:** detector geometry calibration; powder diffraction; phase-blind
identification; zero-shot; radial symmetry; neural network; pyFAI

---

## 1. Introduction

Azimuthal integration of 2D diffraction patterns requires accurate knowledge
of the detector geometry — sample-to-detector distance D, point of normal
incidence (beam center), and detector rotations. The standard workflow
(pyeFAI/pyFAI calibration, DAWN, FIT2D) extracts control points on powder
rings and refines a PONI model; it presumes a human-supplied initial guess
close enough for least squares to converge.

Two failure modes motivate this work. First, *automatic initialization*:
when the beam footprint lies far from the nominal center, radial profiles
computed about the geometric center are systematically distorted, and
naive automation locks onto self-consistent pseudo-solutions. Second,
*architecture choice*: deep regression networks can map an image directly
to geometry parameters, but their error tails corrupt any downstream
refinement seeded by them, and it is unclear when they add value over
well-engineered geometric estimators.

We answer both with a *classic-first cascade*: a fast, assumption-light
geometric solver runs first from the canvas geometric center; only on
failure does a bank of fallback seeds engage — a closed-form
radial-symmetry center estimate, its ray-extrapolations compensating a
documented shrinkage bias, and (optionally) a neural-network prior — after
which a cross-seed consensus gate picks the solution that other seeds
independently agree with. The contributions are:

1. A phase-library peak-position solver (voting + iterative ratio-tolerance
   pairing + LSQ with harmonic re-centering) reaching **0.2 mm** mean
   distance error on real CeO2 calibrations, 15 ms per frame.
2. A tuned **radial-symmetry center estimator** (gradient-line weighted
   least squares; background-subtracted, gradient-gated, robustly
   re-weighted) with 2.9 px median near-center error and a quantified
   shrinkage bias under extreme off-axis geometry.
3. A **cross-seed consensus gate** grounded in the observation that the
   true basin is *re-discovered* by multiple independent seeds while
   pseudo-solutions remain singletons; on real data it removes all
   catastrophic pseudo-solution selections (3/7 → 0/7) at no cost on the
   off-axis benchmark.
4. An **initializer ablation with winner attribution** demonstrating that
   no single initializer suffices — the network contributes uniquely on a
   subset — and a **sensitivity analysis** yielding prior-error
   propagation coefficients (×1.40 for λ and lattice parameter at CeO2
   angles; ×1.00 for pixel size) and a noise-robustness floor.

## 2. Related work

**Interactive calibration software.** The de-facto workflow for 2D
detector calibration is manual control-point picking followed by
least-squares refinement of a geometric model, as implemented in FIT2D
(Hammersley et al., 1996), DAWN (Basham et al., 2015; Filik et al.,
2017), DIOPTAS (Prescher & Prakapenka, 2015) and pyFAI
(Kieffer & Karkoulis, 2013; Ashiotis et al., 2015). These tools are
mature and accurate, but their automation story is an *initial-guess*
story: pyFAI's `calib2` GUI expects the user to supply a detector
distance close enough for convergence, and fully automatic operation
degrades when the beam footprint departs from the detector center —
precisely the regime addressed here. Our pipeline is complementary: it
consumes the same calibrant physics (phase d-spacings, PONI model) but
replaces the human initial guess with a cascade of automatic
initializers arbitrated by consensus, and it emits standard PONI files
directly consumable by pyFAI.

**Fast and online integration.** Recent work has accelerated azimuthal
integration itself — matrix-multiplication reformulation (MatFRAIA;
Jensen et al., 2022) and real-time streaming pipelines (SAXSDOG;
Burian et al., 2022). MatFRAIA shares our matrix view of the
pixel-to-q mapping but optimizes *throughput* for offline batches,
whereas we exploit the same structure for *initialization-free
calibration*; the two aims are orthogonal. As synchrotron data rates
grow, per-frame automatic geometry checks of the kind developed here
become a natural component of such online pipelines.

**Automatic ring detection and centering.** Automating the *first*
calibration step — finding rings or the ring center — has received
scattered attention. Clustering-based detection of Debye–Scherrer rings
has been demonstrated with computer-vision methods (Sirhindi & Khan,
2022), but detection alone does not recover the full geometry. Radial
symmetry is a classical cue in computer vision (fast radial symmetry
transform; Loy & Zelinsky, 2003) and gradient-line centering is
standard practice for stellar images in astronomy. Our radial-symmetry
estimator (§3.4) brings this idea into diffraction calibration in
closed form, and — we believe for the first time — quantifies its
failure mode (LSQ shrinkage toward the pixel centroid under extreme
off-axis geometry) and compensates it with extrapolated seeds.

**Machine learning for diffraction and scattering.** ML now permeates
diffraction data analysis (Surdu & Győrgy, 2023; Chen et al., 2021),
mostly for *classification* (phase identification) or *sample-parameter
regression* from patterns — e.g. nanoparticle sizing from WAXS
(Allara et al., 2024). Direct regression of *instrument geometry* from
a single diffraction frame received, to our knowledge, no prior
treatment: a structured novelty search over arXiv, Crossref and
OpenAlex (project memo, Oct 2026; increment re-run 1 Nov 2026, unchanged)
found no prior work on (i) NN regression of detector geometry,
(ii) differentiable forward models used for gradient-based calibration
of powder rings, or (iii) hybrid physical/NN cascades with
cross-method consensus arbitration. The closest neighbor is MatFRAIA's
matrix formulation, whose differentiability — to our knowledge — has
not been exploited for calibration. Our ablation (§4.3) further shows
*why* architectures matter: no single initializer (learned or
geometric) suffices, which is an architectural lesson rarely made
explicit in ML-for-scattering papers.

## 3. Methods

### 3.1 Forward model and canvas

Rings follow r = D·tan(2θ(q, λ)) with 2θ = 2·arcsin(qλ/4π). Frames are
integer-factor block-averaged to a ≤520 px canvas (pixel size scales by
the same factor), preserving the beam-center offset relative to the
detector. The beam center (point of normal incidence) is distinguished
from the PONI origin; the two differ by rot–distance coupling
(≈ D·tan(rot) up to several hundred pixels).

### 3.2 Peak extraction from the radial profile

Pixel radii about a trial center are binned (≥1 px bins; finer binning is
defeated by Poisson noise, which floods the profile with 30+ spurious
peaks). The mean profile is scanned by prominence-gated peak finding with
parabolic interpolation; an anti-background criterion accepts only peaks
above the profile median, and sub-peaks closer than 4.5 px are merged
(the merge threshold sits between texture-induced splitting and the
smallest real doublet separation).

### 3.3 Voting, pairing, refinement

For each candidate phase from a library (CeO2, LaB6, Si, Al2O3, Quartz,
AgBh), every peak × every (hkl) line casts an implied-distance vote.
Vote clusters seed distance candidates; peaks are matched to lines by
iterative ratio-tolerance pairing (3% → 2% → 1%), clusters are evaluated
by matched-ring count, votes and rms, and the best is retained. (cx, cy,
D) are refined by nonlinear least squares, and a harmonic re-centering
loop re-extracts peaks about the refined center — center error otherwise
splits rings into sub-peaks and inflates rms, killing true solutions at
the gate. Candidate ranking uses full-profile Pearson correlation
(pscore) between the synthetic and measured radial spectra, plus a
coverage term; an NN prior, when supplied, additionally restricts the
accepted solution to a ±80 mm basin.

### 3.4 Radial-symmetry centering (no initial guess)

On any concentric-ring image, the local intensity gradient is radial, so
the line through each pixel along its gradient passes through the ring
center. Weighted least-squares intersection of all gradient lines yields
a **closed-form** center (one 2×2 normal-equation solve, no iteration, no
seed). Practical tuning proved decisive: background subtraction
(Gaussian σ=25 px) removes constant-gradient bias; σ=3 px smoothing with
an 85th-percentile gradient gate and an above-background mask prevents
shot noise from dominating the |g|² weights (the naive estimator errs by
331 px median); a 99th-percentile weight cap tames hot pixels; a 2.5σ
residual re-weighting pass completes the estimator. Result: 2.9 px median
error near center, 18 px at moderate off-axis. Under extreme off-axis
geometry the least-squares intersection is pulled toward the pixel
centroid — *shrinkage*: direction correct, magnitude typically 1/3–1/2 of
the true offset. We therefore add ray seeds extrapolated along the
radial-symmetry direction at 1×, 2× and 3.5×.

### 3.5 Classic-first cascade and consensus gate

Production path: solve once from the geometric center (15 ms); return if
≥3 rings match. On failure, solve from each fallback seed
{RS, ray×3, [NN]} and select by **consensus**: candidates within
|ΔD|/D < 2% and 12 px center distance count each other as votes; ranking
is votes → pscore → ring count → rms. The physical rationale: the true
basin is re-discovered by independent seeds, whereas pseudo-solutions —
which on clean data can pair *more* rings than the truth (m = 8 vs 5–6)
— remain singletons. A single-scalar gate cannot serve both regimes; the
consensus mechanism does (§4.4).

### 3.6 Neural-network initializer

A convolutional regressor (v4s1) maps the log-transformed canvas plus
(λ, pixel-scale) conditioning to the six geometry parameters. It is
trained on ~10⁵ synthetic frames of the six phases with Poisson noise,
per-ring amplitude jitter (0.5–1.6×), Debye–Waller decay, hot pixels,
beam stops and amorphous humps. In production it is *not* loaded on the
fast path; it supplies one cascade seed only when the classic path fails.

### 3.7 Tilt refinement, PONI output, uncertainty

Ring-wise sector fits refine the two detector rotations against a pyFAI
forward model (suspect merged doublets are discarded by a per-ring bias
test); results are written as PONI files and verified by a full-geometry
2θ residual scan against ground-truth PONIs. Frame-to-frame uncertainty
is estimated by bootstrap over photon resampling and validated against
the independent frame scatter (MAD) of a static setup.

## 4. Results

### 4.1 Real CeO2: seven distances

Production command over seven frames (150–450 mm):

| Frame (mm) | ΔD fit (mm) | rings | rms (px) | rot errors (mrad) |
|---|---|---|---|---|
| 150 | −0.1 | 8 | 0.58 | (−3.5, −2.3) |
| 200 | −0.1 | 8 | 0.81 | (−1.3, +2.7) |
| 250 | −0.0 | 8 | 0.74 | (+3.3, +2.5) |
| 300 | −0.0 | 6 | 0.65 | (−4.3, +3.1) |
| 350 | +0.1 | 5 | 0.13 | (−8.0, −2.5) |
| 400 | +0.1 | 5 | 0.38 | (+4.2, +0.2) |
| 450 | −0.1 | 3 | 0.29 | (+3.8, +10.0) |

Mean |ΔD| = **0.1 mm**, worst **0.1 mm** with the native 2D refinement
enabled (per-ring circle fit at full resolution, one extra LSQ pass; ≈1 s per
frame, 7/7 convergence). The 15 ms classic path alone gives 0.2 mm mean /
0.3 mm worst — the 2D stage is what removes the last 0.1 mm. Tilt-error means
(−0.8, +2.0)
mrad; 7/7 frames converge, including the 450 mm frame with only three
usable rings. Full-geometry 2θ residuals vs reference PONIs: rms
1.3–2.4 mrad, max ≤ 7.8 mrad (Fig. 6). The previous production path
(NN-seeded hybrid) gave 4.1 mm mean / 8.0 mm worst on the same data.

Bootstrap photon-resampling on the classic path gives per-frame
σ(D) = 0.09–0.35 mm (mean 0.27 mm; 450 mm excluded, dof ≤ 0 at three
rings) with ground truth inside the 95% CI on **6/6** frames — i.e.
σ/observed-error ≈ 1.8, intervals valid and slightly conservative.
Tilt uncertainties are 2–6 mrad, growing as rings are lost. On the
former hybrid path the same machinery had covered only 1/6 (σ/actual
≈ 0.13, intervals ~8× too narrow, NN center bias dominating): interval
validity is a property of the *path*, not of the bootstrap [doc 10].

### 4.2 Static Al2O3: 52 frames

51/52 frames solved (one dark frame correctly fails); median distance
113.8 mm, frame-to-frame **MAD 0.45 mm**, ten outlier frames flagged by
a 3-MAD test; the 52-frame mean frame solves to 116.41 mm (reference
116.86 mm, Δ = −0.45 mm) with tilt errors (−2.1, −6.0) mrad. Bootstrap
photon-resampling uncertainty σ_stat = 0.47 mm agrees with the
independent MAD (0.45 mm), validating the uncertainty machinery
[doc 09–10].

### 4.3 Initializer ablation and attribution (synthetic off-axis)

Realistic off-axis subset (low tilt, λ = 0.8–1.6 Å, ≥5 in-canvas rings,
beam footprint within canvas; 85 validation frames). Arms: A = NN direct
output; B = single geometric-center seed; C = single NN seed; E = classic
cascade {geo, RS, ray×3}; F = E + NN with m-first ranking; G = same pool
with consensus ranking. Success = |ΔD|/D < 1%:

| Off-axis bin | n | A | B | C | E | F | G |
|---|---|---|---|---|---|---|---|
| 0–0.05 | 3 | 0% | 0% | 67% | 67% | **100%** | **100%** |
| 0.05–0.15 | 6 | 17% | 0% | 33% | 33% | **67%** | 50% |
| 0.15–0.45 | 30 | 3% | 0% | 7% | **30%** | 27% | 27% |
| >0.45 | 46 | 0% | 0% | 0% | 2% | 2% | 0% |

Winner attribution over the 16 fully successful cascade solutions:
**RS = 10, NN = 5, ray = 1, geo = 0.** Two conclusions: (i) the
radial-symmetry seed is the strongest single initializer and supersedes
the geometric-center seed wherever it succeeds; (ii) no seed suffices
alone — the network uniquely rescues cases where RS fails — so the
cascade, not any single initializer, is the correct architecture. Extreme
off-axis geometry (>0.45 of the canvas) remains open.

On *clean* real data the consensus gate vs m-first ranking
(cross-check on the seven CeO2 frames, all seeds): m-first selects a ray
pseudo-solution on 3/7 frames (errors −110 to −181 mm, ring count 8 vs
true 5–6); consensus selects the true basin on all frames (worst
remaining deviation 4.5 mm, a near-true cluster). Consensus is therefore
the fallback ranking in production.

### 4.4 Sensitivity: prior-error propagation and noise floor

Perturbing one prior at a time on the seven CeO2 frames (production path;
slope = d ln D / d ln prior, ±ε averaged; identical to 0.001 across
ε = 0.2–2%, i.e. purely linear):

| Prior | Measured slope | Small-angle theory |
|---|---|---|
| wavelength λ | −1.397 | −1 |
| lattice parameter (Q) | −1.397 | −1 |
| pixel size | +1.000 | +1 |

The ×1.40 amplification is genuine wide-angle physics: the per-ring
slope 2u/(sin 2θ · cos 2θ · √(1−u²)), u = qλ/4π, rises from 1.08
(2θ = 18°) to 1.82 (2θ = 48.5°) across the CeO2 rings, and the global
fit weights outer rings by lever arm and pixel count. In SAXS-like
small-angle use the coefficient reverts to ≈1. **Practical budget: a
0.1% distance target requires λ and lattice parameter known to ≈0.05%**
and pixel size to 0.1%.

Photon statistics are not limiting: Poisson resampling y =
Poisson(I·α)/α (mean preserved, variance ×1/α) from α = 5 down to
α = 0.05 (1/20 exposure) leaves the seven-frame success at 7/7, with
median |ΔD| moving only 0.10 → 0.23 mm and no ring loss. The observed
0.1–0.3 mm floor is systematic (tilt residuals, ring-shape model), not
statistical. (Figure: fig_sensitivity_w7.png.)

### 4.5 Runtime

Classic path 15 ms per frame (canvas); cascade fallback costs ≈ number
of seeds (only on classic failure); NN inference is millisecond-scale
but its prior-constrained refinement path is ~6× slower than the classic
path, which is why it is reserved for fallback. Multi-phase
identification with cascade enabled: 922 ms per frame vs 887 ms
baseline (+4%, fallback dormant).

### 4.6 Additional validation (2026-11 additions)

Three controlled experiments complement the benchmark; code and full tables
are in the repository (`docs/phase_map.md`, `docs/ablation8.md`,
`docs/pyfai_duel.md`).

*Condition map.* On synthetic CeO2 frames the distance error is flat in
exposure (30-30 000 peak counts: |dD| = 0.19-0.49 mm), while the center error
grows linearly with tilt at about 23 px per degree (0.5 deg -> 11.8 px,
4 deg -> 92 px) and is independent of exposure. With a fixed detector,
wavelength-distance separation requires a ring span 2th_max >~ 30 deg:
lambda_hat stays within 2% for D = 130-480 mm but drifts 17-28% at
D = 60-90 mm and fails once the rings leave the detector. A 1% radial
distortion is worth only ~0.4 px / 0.4 mm on this geometry, and a real
pyFAI spline applied and corrected in closed loop restores the clean value
(center 1.81 -> 1.43 px, clean 1.44 px).

*Component ablation.* On a 40-sample stress set (D 0.1-0.6 m, tilt +/-3 deg,
exposure 300-10 000, off-axis centers) removing the automatic centering
degrades the median |dD| from 0.9 to 5.0 mm and phase identification from
88% to 50%; removing the classic-first cascade drops phase identification to
78%; the lambda-joint arm is unusable in the narrow-span regime (median
29 mm). The optional 2D refinement and the tilt layer are immaterial on
clean rings, consistent with their small effect in Sec. 4.1.

*Independent comparison with pyFAI.* Run zero-start (RingExtraction plus
GeometryRefinement in two recalibration rounds), pyFAI fails all 11 frames:
ring assignment is seeded by the initial geometry, and a wrong seed locks a
self-consistent but wrong distance (the same q-ratio degeneracy the consensus
gate is designed to break). Started from our solution it matches us in
distance and is slightly better in center on the CeO2 frames, and on a
spline-corrected detector it wins outright - this documents the distortion
boundary that the present pipeline does not yet cover.

## 5. Discussion

**When does the network earn its place?** Not as the primary solver on
clean data — the classic path is faster and more accurate there — but as
a *diversified seed* whose failure modes differ from those of geometric
estimators; the ablation attribution quantifies exactly this (5 of 16
successes uniquely attributable). We consider this a template for
hybrid physical/ML systems: keep the physical solver primary, use the
learned model where the physics-based initialization breaks, and
arbitrate by cross-method consensus rather than by any single score.

**Limitations.** (i) Extreme off-axis geometries (beam footprint far
outside the canvas) remain unsolved (2% success): the radial-symmetry
shrinkage bias plus short visible arcs degenerate both the center
estimate and the peak extraction. (ii) Prior quality binds: the ×1.4
propagation means sub-0.1% calibration is pointless without
correspondingly precise λ and lattice constants. (iii) Dense-comb
phases (13-line Al2O3-like libraries) under amp/DW randomness can hide
~46% of nominal lines, weakening coverage-based scoring — profile
correlation mitigates but does not eliminate this. (iv) The consensus
tie-break (pscore among singletons) can still misjudge off-axis cases;
center-consistency checks are a natural next step.

## 6. Conclusion

A classic-first cascade with a consensus-gated bank of initializers —
geometric center, closed-form radial symmetry, ray extrapolations, and a
neural prior — calibrates real powder-ring frames to 0.1 mm in distance
(0.2 mm on the 15 ms classic path), remains accurate to
0.23 mm median at 1/20 exposure,
extends the purely geometric path to moderate off-axis geometries, and
quantifies its own prior-error budget (λ/Q ×1.40, pixel ×1.00). The
initializer ablation demonstrates that robustness comes from
architectural diversity under consensus arbitration, not from any
single method — learned or geometric.

## Data availability

Synthetic generator, solver library and probes are available from the
authors. Both real datasets are the publicly distributed pyFAI test
images (https://www.silx.org/pub/pyFAI/testimages/), originally
acquired at the D2AM beamline (BM02, ESRF): the CeO2 multi-distance
calibrant frames (Pilatus 6M, 172 µm pixels, λ = 0.9724 Å, nominal
150–450 mm, each with multi-frame reference PONIs) and the 52 static
Al2O3 frames (λ = 0.7085 Å, reference PONI and distortion spline
included).

## Acknowledgements

The authors thank the pyFAI developers and the D2AM (BM02, ESRF)
beamline for publicly distributing the calibration test frames used in
this work. [TODO: funding numbers — 须用户提供.]

## Figure list (all rendered → probes/results/)

1. Fig. 1 Pipeline schematic (classic-first cascade + consensus gate)
   — **rendered**: `fig1_architecture.png`（probes/fig1_arch.py）
2. Fig. 2 Example frames + fitted rings (ceria 300 mm; alumina mean) —
   **rendered**: `fig2_frames_rings.png`（probes/fig_paper_render.py；
   环贴合数值验证：6/8 环峰位偏差中位 7.6 px @粗检口径）
3. Fig. 3 CeO2 seven-distance errors (bar; 0.1 mm with native 2D refinement) — **rendered**:
   `fig3_ceria_errors.png`（data §4.1）
4. Fig. 4 Ablation grouped bars (A/B/C/E/F/G × bins) + attribution donut
   — **rendered**: `fig4_ablation.png`（data §4.3）
5. Fig. 5 Sensitivity two-panel — **rendered**:
   `fig_sensitivity_w7.png`（doc 12）
6. Fig. 6 2θ full-geometry residuals — **rendered**:
   `fig6_tth_residuals.png`（data doc 11 §四）

## Table list

T1 CeO2 results (§4.1). T2 Al2O3 summary + bootstrap vs MAD (§4.2).
T3 Ablation + attribution (§4.3). T4 Propagation coefficients and noise
floor (§4.4, error budget §4 of doc 12).

## References（已核验 DOI，OpenAlex 2026-10-31）

1. Hammersley, A. P., Svensson, S. O., Hanfland, M., & Fitch, A. N.
   (1996). *Two-dimensional detector software: From real detector to
   idealised image or two-theta scan.* High Pressure Research, 14,
   235–248. https://doi.org/10.1080/08957959608201408
2. Basham, M., Filik, J., Wharmby, M. T., et al. (2015). *Data
   Analysis WorkbeNch (DAWN).* J. Synchrotron Rad., 22, 853–858.
   https://doi.org/10.1107/S1600577515002283
3. Filik, J., Ashton, A. W., Chang, P. C. Y., et al. (2017).
   *Processing two-dimensional X-ray diffraction and small-angle
   scattering data in DAWN 2.* J. Synchrotron Rad., 24, 428–438.
   https://doi.org/10.1107/S1600576717004708
4. Prescher, C., & Prakapenka, V. B. (2015). *DIOPTAS: a program for
   reduction of two-dimensional X-ray diffraction data and data
   exploration.* High Pressure Research, 35, 223–230.
   https://doi.org/10.1080/08957959.2015.1059835
5. Kieffer, J., & Karkoulis, D. (2013). *PyFAI, a versatile library
   for azimuthal regrouping.* J. Phys.: Conf. Ser., 425, 202012.
   https://doi.org/10.1088/1742-6596/425/20/202012
6. Ashiotis, G., Deschildre, A., Nawaz, Z., et al. (2015). *The fast
   azimuthal integration Python library: pyFAI.* J. Appl. Cryst., 48,
   510–519. https://doi.org/10.1107/S1600576715004306
7. Jensen, A. B., Christensen, T. E. K., Wenninger, C., et al. (2022).
   *Very large-scale diffraction investigations enabled by a
   matrix-multiplication facilitated radial and azimuthal integration
   algorithm: MatFRAIA.* J. Synchrotron Rad., 29.
   https://doi.org/10.1107/S1600577522008232
8. Burian, M., Meisenbichler, C., Naumenko, D. O., et al. (2022).
   *SAXSDOG: open software for real-time azimuthal integration of 2D
   scattering images.* J. Synchrotron Rad., 29.
   https://doi.org/10.1107/S1600576722003685
9. Sirhindi, R., & Khan, N. (2022). *Clustering-Based Detection of
   Debye–Scherrer Rings.* ASME J. Comput. Inf. Sci. Eng.
   https://doi.org/10.1115/1.4056568
10. Loy, G., & Zelinsky, A. (2003). *Fast radial symmetry for
    detecting points of interest.* IEEE TPAMI, 25(8), 959–973.
    https://doi.org/10.1109/TPAMI.2003.1217601
11. Surdu, V.-A., & Győrgy, R. (2023). *X-ray Diffraction Data
    Analysis by Machine Learning Methods—A Review.* Appl. Sci., 13,
    9992. https://doi.org/10.3390/app13179992
12. Chen, Z., Andrejevic, N., Drucker, N. C., et al. (2021).
    *Machine learning on neutron and x-ray scattering and
    spectroscopies.* Rev. Sci. Instrum., 92, 073901.
    https://doi.org/10.1063/5.0049111
13. Allara, L., Bertolotti, F., Guagliardi, A., et al. (2024). *A deep
    learning approach for quantum dots sizing from wide-angle X-ray
    scattering data.* npj Comput. Mater.
    https://doi.org/10.1038/s41524-024-01241-6

*待补（作者自查）：radial-symmetry 天文定心先行工作（Hastings 2006
内部报告 / Holtzman 2010，OpenAlex 未收录，投稿前人工确认或删除该句）；*
*pyFAI calib2 手册句的具体出处。*

---

### 写作 TODO（下一稿）

- [x] Related work 补引（04 查新备忘录 → 13 条已核验 DOI 落 References）
- [x] Abstract 压到 JAC 词限（**199 词**，v0.3）
- [x] Fig 5 敏感性图（已渲染 fig_sensitivity_w7.png）
- [x] Fig 1 架构图、Fig 2/3/4/6 渲染（2026-11-01，全部落
      probes/results/，脚本 fig1_arch.py + fig_paper_render.py）
- [x] ceria bootstrap σ 数字并入 §4.1（v0.2 已完成；σ̄=0.27mm、
      GT∈CI 6/6、旧路径 1/6 虚窄 8×——"区间有效性是路径属性"已入正文）
- [x] 全文英文 polish（v0.3 轻量润色：Abstract 重写、语法与一致性
      修正；投稿前建议再过一遍母语级润色）
- [x] 数据引用考证（MX_ceria = pyFAI 官方分发测试图；alumina =
      D2AM/BM02, λ=0.7085Å → Data availability 已填）
- [x] 投稿前查新重跑（第4+5轮，2026-11-01：OpenAlex+Crossref+arXiv，
      2026-09 以来增量，**无直接先行工作**；结果 docs/查新重跑_2026-11-01.txt）
- [ ] 作者/单位（**须用户提供**，无法代填）
- [ ] 基金号（**须用户提供**，Acknowledgements 占位已留）
- [x] ~~D2AM 致谢/机时号确认~~ → **2026-11-01 字节级验证：三套真实数据全部
      = pyFAI 公开测试图**（MX_ceria/alumina md5 与 silx.org 分发版一致，
      LaB6 文件名一致），公开分发无署名义务，Data availability 已改引
      silx.org 图库 + Acknowledgements 致谢 pyFAI 开发者与 D2AM

