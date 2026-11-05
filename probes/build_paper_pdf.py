# -*- coding: utf-8 -*-
"""论文 md → 科研预印本 PDF（A4/Times/两端对齐/图嵌入原位）。
链路：解析 md（丢弃中文管理段）→ 自包含 HTML（base64 图）→
Edge headless --print-to-pdf。产物：桌面 论文_v0.5_20261105.pdf
用法：python -X utf8 probes\build_paper_pdf.py [md路径] [pdf路径]
"""
import base64
import os
import re
import subprocess
import sys
import tempfile
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "probes", "results")
DESK = os.path.join(os.path.expanduser("~"), "Desktop")
MD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    DESK, "论文_v0.5_20261105.md")
PDF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    DESK, "论文_v0.5_20261105.pdf")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# ---------------------------------------------------------------- 图配置
# (显式图号, 文件, 插入锚点[在它之前插入], 宽度%, caption)——图号与正文引用一致
FIGS = [
    (1, "fig1_architecture.png", "### 3.6", 100,
     "Pipeline architecture: the classic peak-position solver runs first "
     "from the canvas centre; on failure, fallback seeds (radial symmetry, "
     "ray extrapolations, optional NN) are arbitrated by the cross-seed "
     "consensus gate."),
    (2, "fig2_frames_rings.png", "### 4.1", 100,
     "Benchmark frames with fitted rings: (a) CeO2 at 300 mm (8 matched "
     "rings); (b) mean of the 52 static Al2O3 frames."),
    (3, "fig3_ceria_errors.png", "### 4.2", 72,
     "CeO2 seven-distance distance errors with the native 2D refinement: "
     "mean and worst |ΔD| = 0.1 mm."),
    (6, "fig6_tth_residuals.png", "### 4.2", 72,
     "Full-geometry 2θ residuals against the reference PONIs over the "
     "whole detector: rms 1.3–2.4 mrad, max ≤ 7.8 mrad."),
    (4, "fig4_ablation.png", "### 4.4", 100,
     "Initializer ablation on synthetic off-axis frames (success = "
     "|ΔD|/D < 1%) with winner attribution of the 16 cascade successes."),
    (5, "fig_sensitivity_w7.png", "### 4.5", 85,
     "Sensitivity analysis: prior-error propagation slopes (λ/lattice "
     "×1.40, pixel ×1.00) and the photon-noise floor down to 1/20 "
     "exposure."),
    (7, "fig7_condition_map.png", "*Component ablation.*", 100,
     "Condition map on synthetic CeO2: (a) exposure is not limiting; "
     "(b) tilt drives the centre error at ≈23 px per degree, independently "
     "of exposure; (c) λ–D separability requires a wide ring system "
     "(2θmax ≳ 30°); (d) a real pyFAI spline corrected in closed loop "
     "restores the clean accuracy."),
    (8, "fig8_duel_ablation.png", "*Independent comparison with pyFAI.*",
     100,
     "Independent comparison with pyFAI and component ablation: (a, b) "
     "zero-start pyFAI (P0) versus our zero-shot pipeline (T) and "
     "same-start refinement (P1) on 11 frames — grey bands mark "
     "spline-corrected detectors outside the present scope; (c, d) "
     "40-frame stress-set ablation of the auto-centering, cascade, "
     "2D-refinement and joint-λ components."),
]

CSS = """
@page { size: A4; margin: 21mm 20mm 23mm; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", "Nimbus Roman", serif;
       font-size: 10.5pt; line-height: 1.34; text-align: justify;
       hyphens: auto; color: #111; margin: 0; }
h1.doctitle { font-size: 15.5pt; text-align: center; margin: 0 0 6pt;
              line-height: 1.25; }
p.meta { text-align: center; font-size: 9pt; color: #333;
         margin: 0 0 2pt; font-style: italic; }
p.meta b { font-style: normal; }
h2 { font-size: 12pt; margin: 15pt 0 5pt; page-break-after: avoid; }
h3 { font-size: 10.8pt; margin: 11pt 0 4pt; page-break-after: avoid; }
p { margin: 0 0 5pt; }
.abstract p, p.keywords { font-size: 9.6pt; }
table { border-collapse: collapse; font-size: 9pt; margin: 6pt auto 8pt;
        page-break-inside: avoid; }
thead tr { border-top: 1.1pt solid #111;
           border-bottom: 0.5pt solid #111; }
tbody tr:last-child { border-bottom: 1.1pt solid #111; }
th, td { padding: 2.2pt 7pt; text-align: left; }
th { font-weight: bold; }
.fig { text-align: center; margin: 9pt 0 3pt; page-break-inside: avoid; }
.fig img { max-width: 100%; }
.cap { font-size: 8.6pt; line-height: 1.28; margin: 2pt 0 9pt;
       text-align: justify; }
.cap b.capno { font-weight: bold; }
ol.refs { font-size: 9pt; padding-left: 16pt; margin: 4pt 0; }
ol.refs li { margin-bottom: 3pt; }
code { font-family: Consolas, monospace; font-size: 9pt; }
sub { font-size: 72%; }
"""

HTML_SHELL = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
              "<title>preprint</title><style>%s</style></head><body>%s"
              "</body></html>")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"<i>\1</i>", s)
    for f, sub in [("CeO2", "CeO<sub>2</sub>"), ("Al2O3", "Al<sub>2</sub>O<sub>3</sub>"),
                   ("LaB6", "LaB<sub>6</sub>"), ("CO2", "CO<sub>2</sub>"),
                   ("AgBh", "AgBh")]:
        s = re.sub(r"(?<![A-Za-z0-9<])" + f + r"(?![A-Za-z0-9])",
                   sub, s)
    return s


def img_tag(num, fname, width, caption):
    with open(os.path.join(RES, fname), "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return ("<div class='fig'><img src='data:image/png;base64,%s' "
            "style='width:%d%%'></div>"
            "<p class='cap'><b class='capno'>Fig. %d.</b> %s</p>"
            % (b64, width, num, inline(caption)))


def table_html(rows):
    out, head = ["<table>"], True
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        tag = "th" if head else "td"
        out.append("<tr>" + "".join(
            "<%s>%s</%s>" % (tag, inline(c), tag) for c in cells) + "</tr>")
        head = False
    out.append("</table>")
    return "".join(out)


def convert():
    lines = open(MD, encoding="utf-8").read().splitlines()
    title = lines[0].lstrip("# ").strip()
    body, i, refs_mode = [], 1, False
    drop_until_h2 = False  # Figure list / Table list / TODO 段丢弃

    fig_by_anchor = {}
    for num, f, anchor, w, cap in FIGS:
        fig_by_anchor.setdefault(anchor, []).append((num, f, w, cap))

    def flush_figs(line):
        for anchor, lst in list(fig_by_anchor.items()):
            if line.startswith(anchor):
                for n, f, w, cap in lst:
                    body.append(img_tag(n, f, w, cap))
                fig_by_anchor.pop(anchor)
                return

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s or s == "---":
            i += 1
            continue
        if s.startswith("**Draft v") or s.startswith("作者："):
            i += 1
            continue
        if s.startswith("## Figure list") or s.startswith("## Table list"):
            drop_until_h2 = True
            i += 1
            continue
        if s.startswith("### 写作 TODO"):
            break
        if drop_until_h2:
            if s.startswith("## "):
                drop_until_h2 = False
            else:
                i += 1
                continue
        if s.startswith("## References"):
            flush_figs(s)
            body.append("<h2>References</h2><ol class='refs'>")
            refs_mode, i = True, i + 1
            continue
        if refs_mode:
            if s.startswith("*待补"):
                i += 1
                if i < len(lines) and not lines[i].strip():
                    i += 1
                continue
            m = re.match(r"^(\d+)\.\s+(.*)$", s)
            if m:
                body.append("<li>%s" % inline(m.group(2)))
            elif s and not s.startswith("##"):
                body[-1] += " " + inline(s)
            i += 1
            continue
        if s.startswith("## "):
            flush_figs(s)
            body.append("<h2>%s</h2>" % inline(s[3:]))
            if "Abstract" in s:
                body.append("<div class='abstract'>")
                # abstract div 在 Keywords 段后闭合
        elif s.startswith("### "):
            flush_figs(s)
            body.append("<h3>%s</h3>" % inline(s[4:]))
        elif s.startswith("|"):
            flush_figs(s)
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = lines[i].strip()
                if not re.match(r"^\|[\s:|-]+\|$", r):
                    rows.append(r)
                i += 1
            body.append(table_html(rows))
            continue
        elif re.match(r"^\d+\.\s", s):
            flush_figs(s)
            body.append("<ol>")
            while i < len(lines) and lines[i].strip() and \
                    not lines[i].strip().startswith(("|", "#")):
                t = lines[i].strip()
                m = re.match(r"^\d+\.\s+(.*)$", t)
                if m:                      # 新条目
                    body.append("<li>%s</li>" % inline(m.group(1)))
                elif body[-1].startswith("<li>"):   # 续行并入上一条
                    body[-1] = body[-1][:-5] + " " + inline(t) + "</li>"
                i += 1
            body.append("</ol>")
            continue
        elif s.startswith("- "):
            flush_figs(s)
            body.append("<ul>")
            while i < len(lines) and lines[i].strip().startswith("- "):
                body.append("<li>%s</li>" %
                            inline(lines[i].strip()[2:]))
                i += 1
            body.append("</ul>")
            continue
        else:
            flush_figs(s)
            # 段落续行合并
            para = [s]
            i += 1
            while i < len(lines):
                t = lines[i].strip()
                if (not t or t.startswith(("#", "|", "- ", "**Draft"))
                        or re.match(r"^\d+\.\s", t)):
                    break
                para.append(t)
                i += 1
            txt = inline(" ".join(para))
            if txt.startswith("<b>Keywords"):
                body.append("</div><p class='keywords'>%s</p>" % txt
                            if "<div class='abstract'>" in body[-8:]
                            else "<p class='keywords'>%s</p>" % txt)
            else:
                body.append("<p>%s</p>" % txt)
            continue
        i += 1

    if fig_by_anchor:
        for lst in fig_by_anchor.values():      # 锚未命中 → 文末兜底
            for n, f, w, cap in lst:
                body.append(img_tag(f, w, n, cap))
    if "<div class='abstract'>" in "".join(body[-10:]):
        body.append("</div>")
    head = ("<h1 class='doctitle'>%s</h1>"
            "<p class='meta'><b>Authors:</b> [to be completed] · "
            "<b>Affiliations:</b> [to be completed]</p>"
            "<p class='meta'>Manuscript draft v0.5 · 5 November 2026 · "
            "target journal: <i>Journal of Applied Crystallography</i></p>"
            % inline(title))
    return HTML_SHELL % (CSS, head + "".join(body))


def main():
    html = convert()
    tmp = os.path.join(tempfile.gettempdir(), "paper_v05_preprint.html")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("html:", tmp, len(html) // 1024, "KB")
    if os.path.exists(PDF):
        os.remove(PDF)
    cmd = [EDGE, "--headless=new", "--disable-gpu",
           "--no-pdf-header-footer",
           "--user-data-dir=%s" % os.path.join(tempfile.gettempdir(),
                                               "edge_pdf_prof"),
           "--print-to-pdf=%s" % PDF, "file:///" + tmp.replace("\\", "/")]
    p = subprocess.run(cmd, capture_output=True, timeout=120)
    for _ in range(20):
        if os.path.exists(PDF) and os.path.getsize(PDF) > 50000:
            break
        time.sleep(0.5)
    if not os.path.exists(PDF):
        print("PDF FAILED", p.returncode, p.stderr.decode(errors="replace")
              [:500])
        sys.exit(1)
    raw = open(PDF, "rb").read()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    print("pdf:", PDF, os.path.getsize(PDF) // 1024, "KB, ~%d pages" % pages)


if __name__ == "__main__":
    main()
