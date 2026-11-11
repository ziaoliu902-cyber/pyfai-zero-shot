# -*- coding: utf-8 -*-
"""_gh_rel093.py:v0.9.3 Release(AgBh/AgSt 歧义警示)发+传 zip。"""
import json
import os
import subprocess
import sys
import urllib.request

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ci = os.path.join(os.environ["TEMP"], "_gh_c14.txt")
open(ci, "w", newline="\n").write("protocol=https\nhost=github.com\n\n")
out = subprocess.run(["git", "credential", "fill"], stdin=open(ci),
                     capture_output=True, text=True, timeout=20)
os.remove(ci)
tok = [l.split("=", 1)[1].strip() for l in out.stdout.splitlines()
       if l.startswith("password=")][0]
HDR = {"Authorization": f"token {tok}",
       "Accept": "application/vnd.github+json", "User-Agent": "rel"}

body = """AgBh / AgSt harmonic-ambiguity warning (user-reported bug).

**What happened**: an AgBh image (Pilatus1M from the silx benchmark, λ=1.0 Å, D=1583 mm) was identified as **AgSt** — reproduced both λ-blind and with the true λ locked. Root cause: both phases have equal-ratio harmonic ring series (d₁/n); with ≤4 rings, rescaling D by 49.70/58.38 maps one onto the other perfectly, and **locking λ does not break it**. Regression introduced when AgSt entered the phase library (11-06); the 6-phase library identified this image correctly.

**Fix**: results for AgBh/AgSt with ≤4 rings now carry an `ambig` flag naming the twin phase. The wizard P3 shows a stronger warning — twin phase name, its corresponding distance, and the go-back-to-② hint (entering the approximate distance resolves the ambiguity; that mechanism already existed). The winner itself is unchanged — no data can separate the two below ~5 rings; pretending otherwise would be dishonest.

Verified: `_t_agbh_repro.py` (ambig present λ-blind and λ-locked), solver canary and wizard multi-frame smoke unaffected.
"""
req = urllib.request.Request(
    "https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot/releases",
    data=json.dumps({"tag_name": "v0.9.3", "target_commitish": "main",
                     "name": "v0.9.3 — AgBh/AgSt ambiguity warning",
                     "body": body}).encode(),
    headers={**HDR, "Content-Type": "application/json"}, method="POST")
import urllib.error
try:
    rel = json.load(urllib.request.urlopen(req, timeout=60))
except urllib.error.HTTPError as e:
    if e.code == 422:
        req = urllib.request.Request(
            "https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
            "/releases/tags/v0.9.3", headers=HDR)
        rel = json.load(urllib.request.urlopen(req, timeout=60))
        print("release 已存在,复用")
    else:
        print("建 release:", e.code, e.read().decode()[:300])
        sys.exit(1)
print(f"release ✓ {rel['html_url']}")

PXY = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
data = open(r"D:\pyfai-project\dist\DiffBlindWizard-v0.9.3.zip", "rb").read()
rp = requests.post(
    f"https://uploads.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
    f"/releases/{rel['id']}/assets?name=DiffBlindWizard-v0.9.3.zip",
    data=data, headers={**HDR, "Content-Type": "application/zip"},
    timeout=(30, 1800), proxies=PXY)
rp.raise_for_status()
a = rp.json()
print(f"资产 ✓ {a['name']} {a['size']/1e6:.1f}MB")
