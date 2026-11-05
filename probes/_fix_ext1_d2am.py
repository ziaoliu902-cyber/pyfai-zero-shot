# -*- coding: utf-8 -*-
"""把 ext1 总结里的 D2AM 归因从"简并"改到"中心出画布/大倾角"。"""
import io

P = r"D:\pyfai-project\docs\ext1_总结_2026-11-04.md"
with io.open(P, encoding="utf-8") as f:
    lines = f.read().split("\n")

out = []
i = 0
while i < len(lines):
    ln = lines[i]
    out.append(ln)
    if "ID28 g17/g45" in ln:
        out.append("   **D2AM 15 帧（接手复核：束心在画布外 800–1500px、rot2 18–50°）**。")
    if "D2AM 0.4959Å（D 塌缩 70%）" in ln:
        # 这一行删掉（D2AM 不再归简并），行首若是顿号也一并清掉
        out[-1] = ln.replace("D2AM 0.4959Å（D 塌缩 70%）、", "").replace("D2AM 0.4959Å（D 塌缩 70%）", "")
    i += 1

txt = "\n".join(out)
txt = txt.replace(
    "mar3450。\n",
    "mar3450。（D2AM 原列此类，接手复核后改归第 1 类：真因是束心出画布，不是 λ–D 简并。）\n", 1)
with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(txt)
print("ok")
