# -*- coding: utf-8 -*-
import io
import sys

P = r"D:\pyfai-project\probes\fig_paper_render.py"
s = io.open(P, encoding="utf-8").read()
a = "mean |ΔD| = 0.2 mm, "
b = "mean |ΔD| = 0.1 mm, "
c = "worst 0.3 mm (15 ms/frame)"
d = "worst 0.1 mm (native 2D refinement)"
assert a in s and c in s, (a in s, c in s)
s = s.replace(a, b, 1).replace(c, d, 1)
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("patched")
