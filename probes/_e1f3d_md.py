# -*- coding: utf-8 -*-
"""_e1f3d_md.py — 当前 md 插 7 条注 + 表头改列名（从 zoo_run 内嵌列表抽取）。"""
import re

MD = r"D:\pyfai-project\docs\zoo_benchmark_v3.md"
ZR = r"D:\pyfai-project\probes\zoo_run.py"

md = open(MD, encoding="utf-8").read()
assert "Δrot(mrad)" in md, "表头缺 Δrot(mrad)"
md = md.replace("Δrot(mrad) | 画布 |", "Δrot/Δc(tilt) | 画布 |", 1)

src = open(ZR, encoding="utf-8").read()
grab = []
for line in src.splitlines():
    t = line.strip()
    if t.startswith('"> 2026-11-03 深夜') and t.endswith('",'):
        grab.append(t[2:-2])
assert len(grab) == 7, f"抽取 {len(grab)} 条"

lines = md.splitlines()
t9 = [k for k, l in enumerate(lines)
      if l.startswith("> 2026-11-03 T9（共心中心真值）")]
assert len(t9) == 1, t9
lines[t9[0] + 1:t9[0] + 1] = ["> " + g for g in grab]
md = "\n".join(lines) + "\n"
open(MD, "w", encoding="utf-8").write(md)
print("完成：注总数", md.count("\n> "))
print("E9M行0.7px:", "Δc(tilt)=0.7px" in md,
      "| E1f3d注:", "E1f3d" in md,
      "| 表头:", "Δrot/Δc(tilt)" in md)
