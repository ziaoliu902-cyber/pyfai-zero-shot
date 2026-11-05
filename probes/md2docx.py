# -*- coding: utf-8 -*-
"""md2docx.py：把进度 md 转成桌面 docx（python-docx，无 pandoc 依赖）。

用法：python -X utf8 probes\\md2docx.py <in.md> <out.docx>
支持：# 标题层级 / 表格（含 \\| 转义、**粗体**、`代码`）/ 引用 / 有无序列表 /
代码块 / --- 分隔线。桌面 docx 与 md 同步用（防再过期）。
"""
import re
import sys

import docx
from docx.shared import Pt

INL = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_runs(p, text):
    for tok in INL.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = p.add_run(tok[2:-2])
            r.bold = True
        elif tok.startswith("`") and tok.endswith("`"):
            r = p.add_run(tok[1:-1])
            r.font.name = "Consolas"
        else:
            p.add_run(tok)


def convert(md_path, docx_path):
    lines = open(md_path, encoding="utf-8").read().splitlines()
    d = docx.Document()
    i, in_code, title_done = 0, False, False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            p = d.add_paragraph()
            r = p.add_run(ln)
            r.font.name = "Consolas"
            r.font.size = Pt(9)
            i += 1
            continue
        s = ln.strip()
        if not s or s == "---":
            i += 1
            continue
        if s.startswith("|"):                          # 表格块
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                t = lines[i].strip().replace("\\|", "\x00")
                cells = [c.strip().replace("\x00", "|")
                         for c in t.strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c or "-")
                           for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                tb = d.add_table(rows=len(rows), cols=len(rows[0]),
                                 style="Table Grid")
                for ri, row in enumerate(rows):
                    for ci, cell in enumerate(row[:len(tb.columns)]):
                        cp = tb.rows[ri].cells[ci].paragraphs[0]
                        add_runs(cp, cell)
                        if ri == 0:
                            for r in cp.runs:
                                r.bold = True
                d.add_paragraph()
            continue
        if s.startswith("#"):
            lvl = len(s) - len(s.lstrip("#"))
            txt = s.lstrip("#").strip()
            if lvl == 1 and not title_done:
                d.add_paragraph(txt, style="Title")
                title_done = True
            else:
                d.add_paragraph(txt, style=f"Heading {min(lvl - 1, 4)}")
        elif s.startswith(">"):
            p = d.add_paragraph(s.lstrip("> ").strip())
            for r in p.runs:
                r.italic = True
        elif re.match(r"^\d+\.\s", s):
            add_runs(d.add_paragraph(style="List Number"),
                     re.sub(r"^\d+\.\s", "", s))
        elif s.startswith(("- ", "* ")) or s == "-":
            indent = 2 if len(ln) - len(ln.lstrip()) >= 2 else 0
            add_runs(d.add_paragraph(style=f"List Bullet {indent + 1}"
                                     if indent else "List Bullet"),
                     s.lstrip("-* ").strip())
        else:
            add_runs(d.add_paragraph(), s)
        i += 1
    d.save(docx_path)
    print(f"OK -> {docx_path}  ({len(d.paragraphs)} 段 / {len(d.tables)} 表)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
