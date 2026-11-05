# -*- coding: utf-8 -*-
"""run_v4s1_night.py：v4s1 夜间全自动链（OS 级保险，不依赖会话存活）。

链：等标签重测完 → 标定复验（不过则停，写报告）→ 冒烟已过跳过 →
训练 v4s1 → _v4s1_check 八档验收 → σq 扫描 → 全部落 probes/results/v4s1_night_report.txt。
"""
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
os.chdir(ROOT)
PY = sys.executable
DATA = os.path.join(ROOT, "data", "train_v4w3")
REPORT = os.path.join(ROOT, "probes", "results", "v4s1_night_report.txt")
LOG = os.path.join(ROOT, "probes", "results", "v4s1_night_live.log")


class _Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.s = sys.stdout

    def write(self, s):
        self.s.write(s)
        self.f.write(s)
        return len(s)

    def flush(self):
        self.s.flush()
        self.f.flush()


def run(cmd, timeout_s):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout_s)
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-4000:], flush=True)
    return p.returncode, out


def main():
    t0 = time.time()
    # ---- 1. 等标签重测（sigq_meta.json 携带新 mtime；标签进程独立在跑）----
    print("等 sigq 重测收尾…", flush=True)
    for _ in range(3600):                      # 最多 1h
        mt = os.path.getmtime(os.path.join(DATA, "sigq.npy"))
        # 2048 版比冒烟晚起跑（00:2x 之后）；超过 25 分钟没更新视为仍在写
        if time.time() - mt < 60:
            break
        time.sleep(30)
    print(f"sigq.npy mtime={time.strftime('%H:%M:%S', time.localtime(mt))}", flush=True)

    # ---- 2. 标定复验：动态范围不过关 → 停 ----
    rc, out = run([PY, "-X", "utf8", "probes\\_sigq_calib.py"], 600)
    lines = [l for l in out.splitlines() if l.strip()]
    sim_ws = [l for l in lines if "0.006" in l or "0.060" in l]
    print("标定关键行:", *sim_ws, sep="\n  ", flush=True)
    # 校验：两档 w 之比 >1.6 才算有动态范围（0.006 vs 0.060 十倍 σq）
    import re
    ws = []
    for l in lines:
        m = re.search(r"0\.0(06|60)\s+([\d.]+)", l)
        if m:
            ws.append(float(m.group(2)))
    if len(ws) >= 2 and min(ws) > 0:
        ratio = max(ws) / min(ws)
    else:
        ratio = 0.0
    print(f"标定动态范围比 = {ratio:.2f}（门槛 1.6）", flush=True)
    if ratio < 1.6:
        print("!! 标定不过关，链条终止——报告已写，等人决断", flush=True)
        return

    # ---- 3. 训练 v4s1 ----
    rc, _ = run([PY, "-X", "utf8", "train.py",
                 "--data", "data/train_v4w3", "--tag", "v4s1",
                 "--init-from", "runs/v4w3/best.pt",
                 "--lam-cond", "--px-cond", "--sig-head",
                 "--epochs", "5", "--batch", "32", "--lr", "5e-5"], 14400)
    if rc != 0:
        print("!! 训练异常退出", flush=True)
        return

    # ---- 4. 八档验收 + σq 扫描 ----
    run([PY, "-X", "utf8", "probes\\_v4s1_check.py"], 900)
    run([PY, "-c",
         "import io;p=r'probes\\_sigma_sweep.py';s=io.open(p,encoding='utf-8').read();"
         "io.open(p,'w',encoding='utf-8').write(s.replace('v4w3','v4s1'))"], 60)
    run([PY, "-X", "utf8", "probes\\_sigma_sweep.py"], 900)

    print(f"\n链条完成，总耗时 {(time.time()-t0)/60:.0f} 分钟", flush=True)


if __name__ == "__main__":
    sys.stdout = _Tee(LOG)
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"!! 链条异常：{e}", flush=True)
