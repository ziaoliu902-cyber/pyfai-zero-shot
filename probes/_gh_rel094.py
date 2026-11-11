# -*- coding: utf-8 -*-
"""_gh_rel094.py:v0.9.4 Release(常见性先验)发+传 zip。"""
import json
import os
import subprocess
import sys
import urllib.request

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ci = os.path.join(os.environ["TEMP"], "_gh_c15.txt")
open(ci, "w", newline="\n").write("protocol=https\nhost=github.com\n\n")
out = subprocess.run(["git", "credential", "fill"], stdin=open(ci),
                     capture_output=True, text=True, timeout=20)
os.remove(ci)
tok = [l.split("=", 1)[1].strip() for l in out.stdout.splitlines()
       if l.startswith("password=")][0]
HDR = {"Authorization": f"token {tok}",
       "Accept": "application/vnd.github+json", "User-Agent": "rel"}

body = """Prevalence prior for the AgBh/AgSt ambiguity (user-approved follow-up to v0.9.3).

AgBh is the de-facto low-q standard (silx benchmark images, NIST-traceable calibrations, most publications); AgSt is niche. When the two fit nearly equally well — which is guaranteed below ~4 rings, their harmonic series are mutually degenerate — the solver now flips to **AgBh** at both arbitration points (λ-blind and λ-locked).

The twin-phase ⚠ hint stays: if your sample really is AgSt, the wizard tells you the corresponding distance — enter it in step ② and the ambiguity resolves. True AgSt data wins by a wider margin and is never overridden.

Verified: Pilatus1M (AgBh, GT λ=1.0 Å) now identifies AgBh on both paths (was AgSt); canary + wizard + P4-advice smokes green.
"""
req = urllib.request.Request(
    "https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot/releases",
    data=json.dumps({"tag_name": "v0.9.4", "target_commitish": "main",
                     "name": "v0.9.4 — prevalence prior (AgBh preferred)",
                     "body": body}).encode(),
    headers={**HDR, "Content-Type": "application/json"}, method="POST")
import urllib.error
try:
    rel = json.load(urllib.request.urlopen(req, timeout=60))
except urllib.error.HTTPError as e:
    if e.code == 422:
        req = urllib.request.Request(
            "https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
            "/releases/tags/v0.9.4", headers=HDR)
        rel = json.load(urllib.request.urlopen(req, timeout=60))
        print("release 已存在,复用")
    else:
        print("建 release:", e.code, e.read().decode()[:300])
        sys.exit(1)
print(f"release ✓ {rel['html_url']}")

PXY = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
data = open(r"D:\pyfai-project\dist\DiffBlindWizard-v0.9.4.zip", "rb").read()
rp = requests.post(
    f"https://uploads.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
    f"/releases/{rel['id']}/assets?name=DiffBlindWizard-v0.9.4.zip",
    data=data, headers={**HDR, "Content-Type": "application/zip"},
    timeout=(30, 1800), proxies=PXY)
rp.raise_for_status()
a = rp.json()
print(f"资产 ✓ {a['name']} {a['size']/1e6:.1f}MB")
