# -*- coding: utf-8 -*-
"""v0.5 §4.1：表格换成 r2d 口径（0.1 mm）。"""
import io

P = r"D:\pyfai-project\paper\论文初稿_phaseblind_zeroshot_v0.5_1105.md"
s = io.open(P, encoding="utf-8").read()

old_table = """| Frame (mm) | ΔD fit (mm) | rings | rms (px) | rot errors (mrad) |
|---|---|---|---|---|
| 150 | −0.2 | 8 | 0.58 | (−4.7, −5.1) |
| 200 | −0.3 | 8 | 0.81 | (−1.9, +0.9) |
| 250 | −0.2 | 8 | 0.74 | (+1.4, +2.5) |
| 300 | +0.0 | 6 | 0.65 | (+2.8, +3.8) |
| 350 | +0.1 | 5 | 0.13 | (−8.0, −2.5) |
| 400 | +0.1 | 5 | 0.38 | (+4.2, +0.2) |
| 450 | −0.1 | 3 | 0.29 | (+3.8, +10.0) |"""

new_table = """| Frame (mm) | ΔD fit (mm) | rings | rms (px) | rot errors (mrad) |
|---|---|---|---|---|
| 150 | −0.1 | 8 | 0.58 | (−3.5, −2.3) |
| 200 | −0.1 | 8 | 0.81 | (−1.3, +2.7) |
| 250 | −0.0 | 8 | 0.74 | (+3.3, +2.5) |
| 300 | −0.0 | 6 | 0.65 | (−4.3, +3.1) |
| 350 | +0.1 | 5 | 0.13 | (−8.0, −2.5) |
| 400 | +0.1 | 5 | 0.38 | (+4.2, +0.2) |
| 450 | −0.1 | 3 | 0.29 | (+3.8, +10.0) |"""

assert old_table in s
s = s.replace(old_table, new_table, 1)

old_txt = """Mean |ΔD| = **0.2 mm**, worst **0.3 mm**. Enabling the optional native 2D
refinement (per-ring circle fit at full resolution, one extra LSQ pass)
lowers this to **0.1 mm mean / 0.1 mm worst** at ≈1 s per frame with the
same 7/7 convergence; all production numbers in this paper are quoted for
the 15 ms classic path unless stated otherwise. Tilt-error means (−0.3, +1.4)"""

new_txt = """Mean |ΔD| = **0.1 mm**, worst **0.1 mm** with the native 2D refinement
enabled (per-ring circle fit at full resolution, one extra LSQ pass; ≈1 s per
frame, 7/7 convergence). The 15 ms classic path alone gives 0.2 mm mean /
0.3 mm worst — the 2D stage is what removes the last 0.1 mm. Tilt-error means
(−0.8, +2.0)"""

assert old_txt in s
s = s.replace(old_txt, new_txt, 1)

# 摘要：主数字改成 0.1（2D），括号里保留经典口径
old_abs = """0.2 mm mean distance error (0.3 mm worst) in 15 ms per frame (0.1 mm with an
optional native 2D refinement), with full-geometry"""
new_abs = """0.1 mm mean distance error (0.1 mm worst) with a native 2D refinement,
0.2 mm (0.3 mm worst) on the 15 ms classic path, with full-geometry"""
assert old_abs in s
s = s.replace(old_abs, new_abs, 1)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("ok")
