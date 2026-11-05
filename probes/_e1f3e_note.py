# -*- coding: utf-8 -*-
"""_e1f3e_note.py — E1f3e 战果补进审计五结论（探针 md 列表 + 当前 md）。"""
P_PY = r"D:\pyfai-project\probes\_nc_audit2.py"
P_MD = r"D:\pyfai-project\docs\nc_audit_v0_1.md"

OLD = "我们的几何常比 GT 更贴数据。"
ADD = ("我们的几何常比 GT 更贴数据；E1f3e（2026-11-04）SolveWorker 原生"
       "优先 tf 移植后生产路径 Eiger4M Δc(V3) 9.32→0.32px、v2 5.62→0.83px"
       "（本表为移植前口径，zoo 表 E1f3d 已是原生口径）。")

src = open(P_PY, encoding="utf-8").read()
n = src.count(OLD)
assert n == 1, f"py 中 {n} 处"          # md 列表（docstring 换行断词不匹配）
src = src.replace(OLD, ADD)
open(P_PY, "w", encoding="utf-8").write(src)
print("py 双处替换 ok")

md = open(P_MD, encoding="utf-8").read()
n = md.count(OLD)
assert n == 1, f"md 中 {n} 处"
md = md.replace(OLD, ADD)
open(P_MD, "w", encoding="utf-8").write(md)
print("md 替换 ok")
