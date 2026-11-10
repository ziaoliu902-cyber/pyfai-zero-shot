# -*- coding: utf-8 -*-
"""_gh_upload_wiz.py:向导版 zip 补传 v0.9.0 Release(2026-11-08)。"""
import json
import os
import subprocess
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ci = os.path.join(os.environ["TEMP"], "_gh_cred3.txt")
open(ci, "w", newline="\n").write("protocol=https\nhost=github.com\n\n")
out = subprocess.run(["git", "credential", "fill"], stdin=open(ci),
                     capture_output=True, text=True, timeout=20)
os.remove(ci)
tok = [l.split("=", 1)[1].strip() for l in out.stdout.splitlines()
       if l.startswith("password=")][0]
HDR = {"Authorization": f"token {tok}",
       "Accept": "application/vnd.github+json", "User-Agent": "pyfai-rel"}

req = urllib.request.Request(
    "https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
    "/releases/tags/v0.9.0", headers=HDR)
rel = json.load(urllib.request.urlopen(req, timeout=60))
print(f"release id={rel['id']}")

data = open(r"D:\pyfai-project\dist\DiffBlindWizard-v0.9.zip", "rb").read()
up = (f"https://uploads.github.com/repos/ziaoliu902-cyber/"
      f"pyfai-zero-shot/releases/{rel['id']}/assets"
      f"?name=DiffBlindWizard-v0.9.zip")
import requests
rp = requests.post(up, data=data, headers={
    "Authorization": f"token {tok}",
    "Content-Type": "application/zip",
    "User-Agent": "pyfai-rel"}, timeout=(30, 1800))
rp.raise_for_status()
a = rp.json()
print(f"上传 ✓ {a['name']} {a['size']/1e6:.1f}MB")

req = urllib.request.Request(rel["assets_url"], headers=HDR)
for x in json.load(urllib.request.urlopen(req, timeout=60)):
    print(" -", x["name"], f"{x['size']/1e6:.1f}MB")
