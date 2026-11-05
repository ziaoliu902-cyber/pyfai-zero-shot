# -*- coding: utf-8 -*-
r"""Zenodo 源码包组装：代码(torchfai+probes) + 生成器(gen_/merge_/train)
+ 图表脚本(fig*) + 证据表(results 文本) + docs + 手稿 + README/LICENSE。
排除 data/ runs/ __pycache__ GUI 运行时杂物。产物 dist\zenodo_code_v0.5.zip
用法：python -X utf8 probes\package_zenodo.py
"""
import compileall
import os
import shutil
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(os.environ.get("TEMP", "/tmp"), "zenodo_stage",
                     "pyfai-zero-shot-v0.5")
OUT = os.path.join(ROOT, "dist", "zenodo_code_v0.5.zip")

README = """# pyfai-zero-shot (v0.5)

Phase-blind, zero-shot geometry calibration of two-dimensional X-ray
detectors from a single powder-ring frame — no calibrant identity, no
initial geometry, and (in joint mode) no wavelength. Outputs a standard
pyFAI PONI plus bootstrap uncertainties.

## Layout

    torchfai/          solver library (phases, simulator, refine, model)
    probes/            benchmark probes, figure scripts, GUI solver
      results/         full result tables (.md/.json/.txt) and reference
                       PONIs backing every number in the paper
    generators/        NN-training data generators + merge tools
    docs/              technical evidence documents (experiment logs)
    manuscript/        paper draft (Markdown, v0.5)

## Key entry points

    probes/solver_core.py    unified solve_image() entry (canvas 520 +
                             zero-shot-safe fallback)
    probes/zoo_run.py        cross-instrument benchmark (silx test images)
    probes/phase_map.py      condition-map experiment (exposure/tilt/span)
    probes/ablation8.py      component ablation (stress set)
    probes/_pyfai_duel.py    independent comparison with pyFAI
    probes/fig_paper_render.py, fig1_arch.py, fig7_8_section46.py
                            all paper figures
    generators/train.py     NN initializer training (synthetic frames)

## Reproduce the benchmark

1. `pip install numpy scipy matplotlib pyFAI fabio pillow h5py torch`
2. Download the public test images referenced in the paper
   (https://www.silx.org/pub/pyFAI/testimages/) into `data/real/`.
3. Run the probes above; every table in `probes/results/` is
   regenerated deterministically (seeds fixed).

## Data

No data are bundled. The real frames are the publicly distributed pyFAI
test images (silx.org, D2AM/BM02 ESRF). Synthetic frames are generated
by `torchfai/simulator.py` + `generators/`.

## License

MIT (see LICENSE). The manuscript draft is provided for transparency.

## Citation

[DOI placeholder — filled at deposition time]
"""

LICENSE = """MIT License

Copyright (c) 2026 The pyfai-zero-shot authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


def cp_tree(src, dst, exts=None, skip_pycache=True):
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        if skip_pycache and "__pycache__" in dirpath:
            continue
        rel = os.path.relpath(dirpath, src)
        for fn in filenames:
            if exts and not fn.lower().endswith(tuple(exts)):
                continue
            d = os.path.join(dst, rel) if rel != "." else dst
            os.makedirs(d, exist_ok=True)
            shutil.copy2(os.path.join(dirpath, fn), os.path.join(d, fn))
            n += 1
    return n


def main():
    if os.path.exists(os.path.dirname(STAGE)):
        shutil.rmtree(os.path.dirname(STAGE))
    os.makedirs(STAGE)
    log = []

    log.append("torchfai: %d files" % cp_tree(
        os.path.join(ROOT, "torchfai"), os.path.join(STAGE, "torchfai"),
        exts=(".py",)))
    log.append("probes py: %d files" % cp_tree(
        os.path.join(ROOT, "probes"), os.path.join(STAGE, "probes"),
        exts=(".py",)))
    log.append("results text: %d files" % cp_tree(
        os.path.join(ROOT, "probes", "results"),
        os.path.join(STAGE, "probes", "results"),
        exts=(".md", ".txt", ".json", ".poni")))
    log.append("docs: %d files" % cp_tree(
        os.path.join(ROOT, "docs"), os.path.join(STAGE, "docs"),
        exts=(".md", ".txt")))

    # 生成器/训练器（根目录精选）
    gen_dst = os.path.join(STAGE, "generators")
    os.makedirs(gen_dst)
    import glob
    pats = ["gen_*.py", "merge_*.py", "train.py", "train_live.py",
            "eval.py", "plot_curves.py", "run_v4s1_night.py"]
    picked = []
    for p in pats:
        picked += glob.glob(os.path.join(ROOT, p))
    for f in picked:
        shutil.copy2(f, os.path.join(gen_dst, os.path.basename(f)))
    log.append("generators: %d files" % len(picked))

    # 手稿
    mdst = os.path.join(STAGE, "manuscript")
    os.makedirs(mdst)
    paper = os.path.join(os.path.expanduser("~"), "Desktop",
                         "论文_v0.5_20261105.md")
    if os.path.exists(paper):
        shutil.copy2(paper, os.path.join(mdst, "paper_v0.5.md"))
        log.append("manuscript: 1 file")

    with open(os.path.join(STAGE, "README.md"), "w",
              encoding="utf-8") as fh:
        fh.write(README)
    with open(os.path.join(STAGE, "LICENSE"), "w", encoding="utf-8") as fh:
        fh.write(LICENSE)

    # 语法体检（不通过就别打 zip）
    ok = compileall.compile_dir(STAGE, quiet=2, force=True)
    shutil.rmtree(os.path.join(STAGE, "__pycache__"), ignore_errors=True)
    for d, _, fs in os.walk(STAGE):
        if "__pycache__" in d:
            shutil.rmtree(d, ignore_errors=True)
    if not ok:
        print("COMPILE CHECK FAILED — abort")
        sys.exit(1)
    log.append("compileall: OK")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, filenames in os.walk(STAGE):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                z.write(full, os.path.relpath(full, os.path.dirname(STAGE)))
    n = len(zipfile.ZipFile(OUT).namelist())
    print(" | ".join(log))
    print("zip:", OUT, os.path.getsize(OUT) // 1024, "KB,", n, "entries")


if __name__ == "__main__":
    main()
