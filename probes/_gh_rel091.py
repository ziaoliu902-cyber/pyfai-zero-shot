# -*- coding: utf-8 -*-
"""_gh_rel091.py:v0.9.1 Release(多帧向导)发+传 zip(2026-11-08)。"""
import json
import os
import subprocess
import sys
import urllib.request

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ci = os.path.join(os.environ["TEMP"], "_gh_c10.txt")
open(ci, "w", newline="\n").write("protocol=https\nhost=github.com\n\n")
out = subprocess.run(["git", "credential", "fill"], stdin=open(ci),
                     capture_output=True, text=True, timeout=20)
os.remove(ci)
tok = [l.split("=", 1)[1].strip() for l in out.stdout.splitlines()
       if l.startswith("password=")][0]
HDR = {"Authorization": f"token {tok}",
       "Accept": "application/vnd.github+json", "User-Agent": "rel"}

body = """Wizard multi-frame mode + confidence tiers + CI.

**New**
- 🧙 **Wizard drops multiple images** (≥2 → multi-frame mode): auto mean seed + per-frame control points + joint least-squares — the full weak-diffraction pipeline now runs inside the 5-step wizard. Phase dropdown + mandatory λ. P4 shows inter-frame MAD consistency lamp. P5 exports joint PONI.
- 🚦 Single-frame confidence tiers (m/ps-based honest labelling; low tier suggests multi-frame re-check)
- 🤖 GitHub Actions CI (syntax gate + guidance smoke + wizard package selftest) — already caught a stale torchfai/phases.py in the repo
- 📋 docs/route_map.md: which route for which data (brightness / angular span / frame count)
- 🧪 Negative result archived: adding a k3/k5 radial-distortion basis to the joint forward makes D *worse* (k–D degeneracy) — explicit spline handling stays

**Downloads (Windows, no Python)**
- 🧙 `DiffBlindWizard-v0.9.1.zip` — 5-step wizard, now multi-frame
- 🛠 `PyFAI-Blind-v0.6.zip` — expert window (batch/joint/drift)
"""
req = urllib.request.Request(
    "https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot/releases",
    data=json.dumps({"tag_name": "v0.9.1", "target_commitish": "main",
                     "name": "v0.9.1 — wizard multi-frame + confidence tiers",
                     "body": body}).encode(),
    headers={**HDR, "Content-Type": "application/json"}, method="POST")
import urllib.error
try:
    rel = json.load(urllib.request.urlopen(req, timeout=60))
except urllib.error.HTTPError as e:
    if e.code == 422:      # 已存在 → 复用
        req = urllib.request.Request(
            "https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
            "/releases/tags/v0.9.1", headers=HDR)
        rel = json.load(urllib.request.urlopen(req, timeout=60))
        print("release 已存在,复用")
    else:
        print("建 release:", e.code, e.read().decode()[:300])
        sys.exit(1)
print(f"release ✓ {rel['html_url']}")

PXY = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
data = open(r"D:\pyfai-project\dist\DiffBlindWizard-v0.9.1.zip", "rb").read()
rp = requests.post(
    f"https://uploads.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
    f"/releases/{rel['id']}/assets?name=DiffBlindWizard-v0.9.1.zip",
    data=data, headers={**HDR, "Content-Type": "application/zip"},
    timeout=(30, 1800), proxies=PXY)
rp.raise_for_status()
a = rp.json()
print(f"资产 ✓ {a['name']} {a['size']/1e6:.1f}MB")
