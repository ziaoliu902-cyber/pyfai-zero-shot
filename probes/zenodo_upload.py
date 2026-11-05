# -*- coding: utf-8 -*-
r"""Zenodo 存缴：建 deposition → 传 zip → 填元数据 → (可选)publish 拿 DOI
→ 回填 DOI 到桌面论文 md + 一页纸并重建两个 PDF。
用法：
  set ZENODO_TOKEN=xxx            （zenodo.org → Settings → Applications →
                                   New token，勾 deposit:actions+deposit:write）
  python -X utf8 probes\zenodo_upload.py            # 只建草稿+传文件
  python -X utf8 probes\zenodo_upload.py publish    # 发布并回填 DOI
"""
import json
import os
import re
import subprocess
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, "dist", "zenodo_code_v0.5.zip")
DESK = os.path.join(os.path.expanduser("~"), "Desktop")
MD = os.path.join(DESK, "论文_v0.5_20261105.md")
PLACEHOLDER = "10.5281/zenodo.XXXXXXX"

TOKEN = os.environ.get("ZENODO_TOKEN", "")
BASE = "https://zenodo.org/api"

AUTHORS_FALLBACK = "pyfai-zero-shot developers"   # 无作者信息时的占位
DESCRIPTION = (
    "<p>Phase-blind, zero-shot geometry calibration of 2D X-ray detectors "
    "from a single powder-ring frame: no calibrant identity, no initial "
    "geometry, optional joint wavelength solve. Classic-first cascade "
    "(peak-position solver + radial-symmetry/ray/NN fallback seeds) with a "
    "cross-seed consensus gate; outputs standard pyFAI PONI files with "
    "bootstrap uncertainties.</p>"
    "<p>This archive contains the solver library (torchfai), all benchmark "
    "probes and figure scripts (probes), NN-training data generators, the "
    "full result tables backing every number in the manuscript, and the "
    "manuscript draft v0.5. Real test images are the public pyFAI test "
    "sets (silx.org, D2AM/BM02 ESRF) and are not redistributed "
    "here.</p>")


def main():
    publish = "publish" in sys.argv
    if not TOKEN:
        print("no ZENODO_TOKEN in env — abort")
        sys.exit(1)
    if not os.path.exists(ZIP):
        print("zip missing — run probes/package_zenodo.py first")
        sys.exit(1)
    H = {"Authorization": "Bearer %s" % TOKEN}

    # 1. 新建 deposition
    r = requests.post(BASE + "/deposit/depositions", headers=H, json={},
                      params={"access_token": TOKEN}, timeout=60)
    r.raise_for_status()
    dep = r.json()
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print("deposition:", dep_id, "(draft)")

    # 2. 传文件
    with open(ZIP, "rb") as fh:
        r = requests.put(
            "%s/%s" % (bucket, os.path.basename(ZIP)), headers=H,
            data=fh, timeout=1800)
    r.raise_for_status()
    print("uploaded:", os.path.basename(ZIP),
          os.path.getsize(ZIP) // 1024, "KB")

    # 3. 元数据
    meta = {"metadata": {
        "title": "pyfai-zero-shot v0.5: phase-blind zero-shot geometry "
                 "calibration of 2D X-ray detectors from powder rings "
                 "(solver, generators, benchmark probes, figures)",
        "upload_type": "software",
        "description": DESCRIPTION,
        "creators": [{"name": os.environ.get(
            "ZENODO_AUTHOR", AUTHORS_FALLBACK)}],
        "license": "MIT",
        "keywords": ["X-ray diffraction", "detector geometry calibration",
                     "powder rings", "pyFAI", "zero-shot", "phase-blind"],
    }}
    r = requests.put(
        "%s/deposit/depositions/%s" % (BASE, dep_id),
        headers=dict(H, **{"Content-Type": "application/json"}),
        params={"access_token": TOKEN},
        data=json.dumps(meta), timeout=60)
    r.raise_for_status()
    print("metadata set")

    if not publish:
        print("draft ready: https://zenodo.org/deposit/%d — 网页里补作者/"
              " affiliation 后手动 Publish，或重跑本脚本加 publish" % dep_id)
        return

    # 4. 发布
    r = requests.post("%s/deposit/depositions/%s/actions/publish" %
                      (BASE, dep_id), headers=H,
                      params={"access_token": TOKEN}, timeout=60)
    r.raise_for_status()
    doi = r.json()["doi"]
    print("PUBLISHED doi:", doi)

    # 5. 回填 md + 一页纸脚本 + 重建 PDF
    for path, old, new in [
        (MD, PLACEHOLDER, doi),
        (os.path.join(ROOT, "probes", "build_onepager_pdf.py"),
         "Zenodo DOI [to be assigned]",
         "Zenodo DOI " + doi),
    ]:
        if os.path.exists(path):
            s = open(path, encoding="utf-8").read()
            if old in s:
                open(path, "w", encoding="utf-8").write(s.replace(old, new))
                print("replaced in", os.path.basename(path))
    for script in ("build_paper_pdf.py", "build_onepager_pdf.py"):
        subprocess.run([sys.executable, "-X", "utf8",
                        os.path.join(ROOT, "probes", script)], cwd=ROOT)


if __name__ == "__main__":
    main()
