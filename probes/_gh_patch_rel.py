# -*- coding: utf-8 -*-
"""_gh_patch_rel.py:更新 v0.9.0 Release 说明(双包指引,2026-11-08)。"""
import json
import os
import subprocess
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ci = os.path.join(os.environ["TEMP"], "_gh_cred4.txt")
open(ci, "w", newline="\n").write("protocol=https\nhost=github.com\n\n")
out = subprocess.run(["git", "credential", "fill"], stdin=open(ci),
                     capture_output=True, text=True, timeout=20)
os.remove(ci)
tok = [l.split("=", 1)[1].strip() for l in out.stdout.splitlines()
       if l.startswith("password=")][0]
HDR = {"Authorization": f"token {tok}",
       "Accept": "application/vnd.github+json", "User-Agent": "pyfai-rel"}

body = """Weak-diffraction multi-frame pipeline + joint calibration, zoo hang fix, UX guards. See commit 2076aad for the full changelog.

**Highlights**
- 🧩 Joint calibration (B13): control-point global LSQ, two-layer solver (0.4 µm parity vs reference)
- 🏠 Weak-diffraction mean-frame seed: full-frame mean recovers SNR (√N); 51-frame e2e all green (seed −0.13 mm, σ_boot 0.019 mm, ~2 min)
- 🐛 zoo hang fixed: split-Gauss center 38.5→2.4 ms (15.8×); 10-row suite restored
- 🛟 UX: escalation top-fail → multi-frame guidance (one-click batch); λ–D degeneracy σ-graded warning

**Downloads (Windows, no Python needed — unzip & double-click)**
- 🧙 `DiffBlindWizard-v0.9.zip` — **recommended for beginners**: 5-step wizard (pick image → confirm params → solve → verdict light → export)
- 🛠 `PyFAI-Blind-v0.6.zip` — expert main window (batch mode, joint calibration, weak-diffraction mean-frame seed, drift curves)

Both run `--selftest` for a headless integrity check. See 使用说明.txt inside each zip.
"""
req = urllib.request.Request(
    "https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
    "/releases/439966052",
    data=json.dumps({"body": body}).encode(),
    headers={**HDR, "Content-Type": "application/json"}, method="PATCH")
rel = json.load(urllib.request.urlopen(req, timeout=60))
print("patched ✓", rel["html_url"])
