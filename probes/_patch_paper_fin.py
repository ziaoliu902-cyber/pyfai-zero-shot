# -*- coding: utf-8 -*-
import io

P = r"D:\pyfai-project\paper\论文初稿_phaseblind_zeroshot_v0.5_1105.md"
s = io.open(P, encoding="utf-8").read()

a = "calibrates real powder-ring frames to 0.2 mm in distance\n(0.1 mm with the optional 2D refinement) within 15 ms, remains accurate to"
b = "calibrates real powder-ring frames to 0.1 mm in distance\n(0.2 mm on the 15 ms classic path), remains accurate to"
if a in s:
    s = s.replace(a, b, 1)
    print("conclusion fixed")
else:
    print("conclusion anchor miss")

c = "3. Fig. 3 CeO2 seven-distance errors (bar) — **rendered**:"
d = "3. Fig. 3 CeO2 seven-distance errors (bar; 0.1 mm with native 2D refinement) — **rendered**:"
if c in s:
    s = s.replace(c, d, 1)
    print("fig3 list updated")

io.open(P, "w", encoding="utf-8", newline="").write(s)
