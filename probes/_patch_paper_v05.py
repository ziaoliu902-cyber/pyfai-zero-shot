# -*- coding: utf-8 -*-
"""v0.4 → v0.5：修事实错误 + 并入新证据（条件相图/消融/pyFAI 对照）。"""
import io
import os
import time

P = os.path.join(r"D:\pyfai-project", "paper",
                 "论文初稿_phaseblind_zeroshot_v0.4_1103.md")
s = io.open(P, encoding="utf-8").read()
orig = s

# 1) Abstract：相鉴定口径（8 → 6/8）+ 2D 精化后 0.1mm
s = s.replace(
    "On a ten-image\ncross-instrument benchmark it identifies the phase on all eight evaluable\nframes and recovers the distance within 0.1 mm on four detectors.",
    "On a ten-image\ncross-instrument benchmark it identifies the phase on six of the eight\nevaluable frames and recovers the distance within 0.1 mm on four detectors.")
s = s.replace(
    "the pipeline attains\n0.2 mm mean distance error (0.3 mm worst) in 15 ms per frame",
    "the pipeline attains\n0.2 mm mean distance error (0.3 mm worst) in 15 ms per frame, 0.1 mm with an\noptional native 2D refinement")

# 2) §4.1：补一句 2D 精化
s = s.replace(
    "Mean |ΔD| = **0.2 mm**, worst **0.3 mm**; tilt-error means (−0.3, +1.4)",
    "Mean |ΔD| = **0.2 mm**, worst **0.3 mm**. Enabling the optional native 2D\n"
    "refinement (per-ring circle fit at full resolution, one extra LSQ pass)\n"
    "lowers this to **0.1 mm mean / 0.1 mm worst** at ≈1 s per frame with the\n"
    "same 7/7 convergence; all production numbers in this paper are quoted for\n"
    "the 15 ms classic path unless stated otherwise. Tilt-error means (−0.3, +1.4)")

# 3) §4.4 噪声地板
s = s.replace("The observed\n0.2–0.3 mm floor", "The observed\n0.1–0.3 mm floor")

# 4) Conclusion：补 0.1mm
s = s.replace(
    "calibrates real powder-ring frames to 0.2 mm in distance\nwithin 15 ms, remains accurate to 0.23 mm median at 1/20 exposure",
    "calibrates real powder-ring frames to 0.2 mm in distance\n(0.1 mm with the optional 2D refinement) within 15 ms, remains accurate to\n0.23 mm median at 1/20 exposure")

# 5) 新增 §4.6
new_sec = """### 4.6 Additional validation (2026-11 additions)

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

"""
s = s.replace("## 5. Discussion", new_sec + "## 5. Discussion", 1)

assert s != orig, "no replacements applied"
out = os.path.join(r"D:\pyfai-project", "paper",
                   "论文初稿_phaseblind_zeroshot_v0.5_1105.md")
io.open(out, "w", encoding="utf-8").write(s)
print("saved", out, os.path.getsize(out))
