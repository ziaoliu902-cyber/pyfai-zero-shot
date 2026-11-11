# -*- coding: utf-8 -*-
"""_gh_cilog.py:拉 CI 失败日志错误行(2026-11-08)。"""
import json
import os
import subprocess
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ci = os.path.join(os.environ["TEMP"], "_gh_c8.txt")
open(ci, "w", newline="\n").write("protocol=https\nhost=github.com\n\n")
out = subprocess.run(["git", "credential", "fill"], stdin=open(ci),
                     capture_output=True, text=True, timeout=20)
os.remove(ci)
tok = [l.split("=", 1)[1].strip() for l in out.stdout.splitlines()
       if l.startswith("password=")][0]
H = {"Authorization": f"token {tok}", "User-Agent": "ci-check"}

req = urllib.request.Request(
    "https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
    "/actions/runs?per_page=2", headers=H)
runs = json.load(urllib.request.urlopen(req, timeout=30))["workflow_runs"]
j = runs[0]                       # 最新一条(默认 main 最近 push)
req = urllib.request.Request(j["jobs_url"], headers=H)
job = json.load(urllib.request.urlopen(req, timeout=30))["jobs"][0]
import requests                                        # 跨域重定向自动丢 auth 头
logs = requests.get(
    f"https://api.github.com/repos/ziaoliu902-cyber/pyfai-zero-shot"
    f"/actions/jobs/{job['id']}/logs", headers=H, timeout=60).text
keep = [l for l in logs.splitlines()
        if any(k in l for k in ("Error", "error", "Traceback", "assert",
                                "FAIL", "Exception"))][:15]
print("\n".join(keep) or "(无错误行,打印尾部)")
if not keep:
    print("\n".join(logs.splitlines()[-25:]))
