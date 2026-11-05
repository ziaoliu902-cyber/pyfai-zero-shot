# -*- coding: utf-8 -*-
r"""外联一页纸 PDF：问题 / 方法一句话 / 证据(6/8 相鉴 + 4/8 亚毫米 + 失败归因)
/ 请求(1–2 张带 poni 的历史图盲测)。Edge headless 打印，产物桌面
一页纸_零样本标定_盲测请求_20261105.pdf。用法：
python -X utf8 probes\build_onepager_pdf.py
"""
import os
import subprocess
import sys
import tempfile
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESK = os.path.join(os.path.expanduser("~"), "Desktop")
PDF = os.path.join(DESK, "一页纸_零样本标定_盲测请求_20261105.pdf")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 15mm 16mm 13mm; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", serif; font-size: 9.8pt;
       line-height: 1.3; color: #111; margin: 0; text-align: justify;
       hyphens: auto; }
.band { background: #1a365d; color: #fff; padding: 7pt 10pt;
        margin: -2pt 0 7pt; }
.band h1 { font-size: 14.5pt; margin: 0 0 1pt; line-height: 1.15; }
.band p { margin: 0; font-size: 9pt; font-style: italic; color: #cbd5e0;
          text-align: left; }
h2 { font-size: 10.8pt; margin: 8pt 0 3pt; color: #1a365d;
     border-bottom: 0.7pt solid #1a365d; padding-bottom: 1pt; }
p { margin: 0 0 3.5pt; }
ul { margin: 0 0 4pt 0; padding-left: 14pt; }
li { margin-bottom: 2.2pt; }
table { border-collapse: collapse; font-size: 9.2pt; margin: 3pt 0 5pt;
        width: 100%; }
thead tr { border-top: 1.1pt solid #111;
           border-bottom: 0.5pt solid #111; }
tbody tr:last-child { border-bottom: 1.1pt solid #111; }
th, td { padding: 2pt 5pt; text-align: left; }
.ask { border: 1.2pt solid #1a365d; background: #ebf4ff; padding: 6pt 9pt;
       margin-top: 2pt; }
.ask p { margin-bottom: 2.5pt; }
.foot { margin-top: 7pt; font-size: 8.2pt; color: #444;
        border-top: 0.5pt solid #999; padding-top: 3pt; }
b.hl { color: #1a365d; }
sub { font-size: 72%; }
"""

BODY = """
<div class="band">
<h1>Zero-shot geometry calibration of 2D diffraction detectors
— a request for two historical frames</h1>
<p>Phase-blind calibration from a single powder image: no calibrant
identity, no initial guess, PONI out &middot; blind-test call, Nov 2026</p>
</div>

<h2>The problem</h2>
<p>Azimuthal integration needs the detector geometry (distance, beam
centre, tilts). Standard tools (pyFAI calib2, DAWN, FIT2D) presume a
<b>known calibrant, manual ring picking and a good initial guess</b>; run
from a cold start they lock onto self-consistent but wrong solutions
(on the same benchmark below, zero-start pyFAI fails ten of eleven
frames, up to 787 mm distance error). Automated, unattended calibration
does not exist in practice.</p>

<h2>Our method, in one sentence</h2>
<p>A classic-first cascade — a fast peak-position solver from the canvas
centre, then closed-form radial-symmetry / ray / optional NN fallback
seeds — pairs profile peaks to a phase library (CeO<sub>2</sub>, LaB<sub>6</sub>,
Si, Al<sub>2</sub>O<sub>3</sub>, quartz, AgBh), refines (c<sub>x</sub>,
c<sub>y</sub>, D) by least squares with a cross-seed <b>consensus gate</b>
that kills pseudo-solutions, and writes a standard PONI with bootstrap
uncertainties.</p>

<h2>Where it stands (public benchmark images, silx.org/pyFAI)</h2>
<table>
<thead><tr><th>Test</th><th>Result</th></tr></thead>
<tbody>
<tr><td>CeO<sub>2</sub> calibrant, 7 distances 150–450 mm</td>
    <td>mean |ΔD| = <b class="hl">0.1 mm</b> (worst 0.1 mm);
        classic path 15 ms/frame</td></tr>
<tr><td>52 static Al<sub>2</sub>O<sub>3</sub> frames</td>
    <td>frame scatter MAD <b class="hl">0.45 mm</b>; bootstrap σ agrees
        (0.47 mm)</td></tr>
<tr><td>Cross-instrument, 10 public images, 8 evaluable</td>
    <td>phase identified <b class="hl">6/8</b>; distance ≤ 0.1 mm
        <b class="hl">4/8</b>; tilt ≲ 5 mrad</td></tr>
<tr><td>Failure attribution (the 2 misses)</td>
    <td>Frelon2k &amp; halfccd — the only two frames whose reference
        calibration carries a <b>distortion spline</b>; our forward model
        has no distortion term (declared boundary, not a solver crash:
        pyFAI with the spline wins there). The two sub-0.1 mm misses sit
        on the λ–D ridge: geometry correct (ΔλD ≤ 0.85%), the fitted
        wavelength absorbs a reference-λ offset.</td></tr>
<tr><td>Independent check</td>
    <td>zero-start pyFAI fails 10/11 on the same frames; started from our
        solution it matches us (validates the answer, not the start)</td></tr>
</tbody>
</table>

<h2>What we ask of you</h2>
<div class="ask">
<p><b>1–2 historical frames with a reference .poni</b> — any detector,
calibrant, distance, wavelength, vintage; distortion splines welcome
(that is exactly the boundary we want to probe). Nothing new to
measure: archived commissioning or calibration images are ideal.</p>
<p><b>Blind protocol:</b> you send the image (plus pixel size; optionally
wavelength or “unknown”); we return distance, beam centre, tilts, a PONI
and uncertainty intervals — plus the phase identification — without
seeing your calibration. We report the comparison back to you either
way, and keep your data unshared on request.</p>
</div>

<p style="margin-top:5pt"><b>Why it is worth 10 minutes:</b> each frame
extends a public, reproducible cross-instrument benchmark; failures are
diagnosed, not hidden (see attribution above), and every result ships
with uncertainty intervals validated against ground truth (6/6 coverage
on the CeO<sub>2</sub> set).</p>

<div class="foot">
Contact: [name · email — to be completed] &nbsp;|&nbsp;
Code &amp; full tables: Zenodo DOI [to be assigned] &nbsp;|&nbsp;
Manuscript draft v0.5 (2026-11-05), target <i>J. Appl. Cryst.</i>
&nbsp;|&nbsp; Test images: silx.org/pub/pyFAI/testimages (D2AM/BM02, ESRF)
</div>
"""

HTML = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>onepager</title><style>%s</style></head><body>%s</body>"
        "</html>" % (CSS, BODY))


def main():
    tmp = os.path.join(tempfile.gettempdir(), "onepager.html")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(HTML)
    if os.path.exists(PDF):
        os.remove(PDF)
    cmd = [EDGE, "--headless=new", "--disable-gpu",
           "--no-pdf-header-footer",
           "--user-data-dir=%s" % os.path.join(tempfile.gettempdir(),
                                               "edge_pdf_prof"),
           "--print-to-pdf=%s" % PDF, "file:///" + tmp.replace("\\", "/")]
    subprocess.run(cmd, capture_output=True, timeout=120)
    for _ in range(20):
        if os.path.exists(PDF) and os.path.getsize(PDF) > 20000:
            break
        time.sleep(0.5)
    if not os.path.exists(PDF):
        print("FAILED")
        sys.exit(1)
    import re
    raw = open(PDF, "rb").read()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    print("pdf:", PDF, os.path.getsize(PDF) // 1024, "KB,", pages, "page(s)")


if __name__ == "__main__":
    main()
