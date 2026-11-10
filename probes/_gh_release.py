# -*- coding: utf-8 -*-
"""_gh_release.py:发 GitHub Release v0.9.0 + 挂傻瓜版 zip(2026-11-08)。

token 走 git credential fill(Windows 凭据管理器,ziaoliu902-cyber PAT);
stdin 必须按行文件重定向(PowerShell 管道会吞换行)。
"""
import json
import os
import subprocess
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- 1) credential fill 拿 token ----
proto = os.environ.get("TEMP", ".")
cred_in = os.path.join(proto, "_gh_cred.txt")
with open(cred_in, "w", newline="\n") as f:
    f.write("protocol=https\nhost=github.com\n\n")
try:
    out = subprocess.run(["git", "credential", "fill"], stdin=open(cred_in),
                         capture_output=True, text=True, timeout=20)
finally:
    os.remove(cred_in)
tok = None
for ln in out.stdout.splitlines():
    if ln.startswith("password="):
        tok = ln.split("=", 1)[1].strip()
if not tok:
    print("credential fill 失败:", out.stdout, out.stderr)
    sys.exit(1)
print(f"token 拿到(长度 {len(tok)})")

API = "https://api.github.com"
HDR = {"Authorization": f"token {tok}",
       "Accept": "application/vnd.github+json",
       "User-Agent": "pyfai-release-script"}

# ---- 2) 建 Release ----
body = """Weak-diffraction multi-frame pipeline + joint calibration, zoo hang fix, UX guards. See commit 2076aad for the full changelog.

**Highlights**
- 🧩 Joint calibration (B13): control-point global LSQ, two-layer solver (0.4 µm parity vs reference)
- 🏠 Weak-diffraction mean-frame seed: full-frame mean recovers SNR (√N); 51-frame e2e all green (seed −0.13 mm, σ_boot 0.019 mm, ~2 min)
- 🐛 zoo hang fixed: split-Gauss center 38.5→2.4 ms (15.8×, tolerance+analytic jac+height gate); 10-row suite restored
- 🛟 UX: escalation top-fail → multi-frame guidance (one-click batch); λ–D degeneracy σ-graded warning
- 📦 **Windows傻瓜版**: download `PyFAI-Blind-v0.6.zip` below, unzip, double-click `PyFAI-Blind.exe` (no Python needed; see 使用说明.txt inside)
"""
req = urllib.request.Request(
    API + "/repos/ziaoliu902-cyber/pyfai-zero-shot/releases",
    data=json.dumps({
        "tag_name": "v0.9.0",
        "target_commitish": "main",
        "name": "v0.9.0 — weak-diffraction pipeline + joint calibration",
        "body": body,
        "prerelease": False,
    }).encode(), headers={**HDR, "Content-Type": "application/json"},
    method="POST")
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        rel = json.load(r)
        print(f"release 创建: id={rel['id']} tag={rel['tag_name']} "
              f"url={rel['html_url']}")
except urllib.error.HTTPError as e:
    print("建 release 失败:", e.code, e.read().decode()[:500])
    sys.exit(1)

# ---- 3) 上传 zip 资产 ----
zip_path = r"D:\pyfai-project\dist\PyFAI-Blind-v0.6.zip"
up = (f"https://uploads.github.com/repos/ziaoliu902-cyber/"
      f"pyfai-zero-shot/releases/{rel['id']}/assets"
      f"?name=PyFAI-Blind-v0.6.zip")
data = open(zip_path, "rb").read()
req = urllib.request.Request(
    up, data=data,
    headers={**HDR, "Content-Type": "application/zip"},
    method="POST")
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        a = json.load(r)
        print(f"资产上传 ✓ {a['name']} {a['size']/1e6:.1f}MB "
              f"下载数={a['download_count']}")
        print(f"\nRelease 页: {rel['html_url']}")
except urllib.error.HTTPError as e:
    print("上传资产失败:", e.code, e.read().decode()[:300])
    sys.exit(1)
