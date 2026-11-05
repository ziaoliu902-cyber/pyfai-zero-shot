# -*- coding: utf-8 -*-
"""pyfai-project 驾驶舱：一个网页看全项目状态。

自动检测（无需登记）：
  - 系统：GPU 利用率/显存、python 进程分类（训练/实验/其他）
  - 训练：最新 runs/*/history.csv → epoch 进度、loss 趋势、best
  - 战果：v2 vs v3 分档、软先验 A/B（静态，更新结论时改 VERDICTS）
  - 待办：当前任务清单
用法：python cockpit.py   →  http://localhost:8765
"""
import csv
import glob
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8765

VERDICTS = [
    {"title": "全量 394 终版：M1cd（标定σ+双轨） vs M1（软先验单轨）",
     "head": ["指标", "M1", "M1cd", "显著性"],
     "rows": [
         ["dist 中位 mm", "0.220", "0.184", "p=2.9e-3 (227/394 胜)"],
         ["<0.5mm 占比", "67.8%", "69.5%", "—"],
         ["rot2 中位 mrad", "8.38", "7.76", "p=4.2e-5 (238/394 胜)"],
         ["双轨采纳率 / 误伤", "—", "21% / 1张>1mm", "零误伤设计验证"]]},
    {"title": "Sim-to-Real：ceria 七档物理层对拍（λ=0.972Å, Pilatus6M）",
     "head": ["档 mm", "150", "200", "250", "300", "350", "400", "450"],
     "rows": [
         ["dist 误差 mm", "-0.02", "+0.12", "+3.65⚠", "+0.61", "-2.03", "+1.15", "-1.06"]],
     },
    {"title": "微调事故链（v4 系列已闭环）",
     "head": ["实验", "结论"],
     "rows": [
         ["v4b '灾难性遗忘'", "meta 编码常数错位（1311mm 假象），精确复现"],
         ["v4c 只修编码", "远距遗忘 +48%"],
         ["v4d +6k 远距重放", "远距 26.7→18.6mm (-30%)，重放有效"],
         ["v4d vs base_v3", "净收益 ≈ 0 → 域差由精修层吸收，微调路线关闭"]]},
]

TODOS = [
    ("done", "训练仓清理 147GB + 磁盘体检（train_v3+real 精简保留）"),
    ("done", "σ 温度标定落盘（rot1/rot2 条件失准 3× 发现→双轨设计）"),
    ("done", "真实数据对拍：物理层七档中位 1.06mm（NN 裸跑失败=λ 简并）"),
    ("done", "论文 NC 初稿 docs/paper_draft_NC_v1.md"),
    ("doing", "λ-conditioned 训练仓 train_lam_v1 生成中（150k，0.5-2.3Å）"),
    ("todo", "train_lam_v1 10/10 审计（λ 分布/越界/行数=分片和）"),
    ("todo", "λ 模型训练（等指令）→ 三波长真实对拍 v3"),
    ("todo", "NC P0：vs pyFAI calib2d 正面对比 + real 库全量泛化"),
]


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=15).stdout
    except Exception:
        return b""


def collect():
    st = {"time": time.strftime("%H:%M:%S")}
    g = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader"]).decode(errors="replace").strip()
    if "," in g:
        st["gpu"] = g.replace("\n", " | ")
    ps = sh(["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR "
             "Name='pythonw.exe'\" | Select-Object ProcessId,CommandLine | "
             "ConvertTo-Json"]).decode(errors="replace")
    procs = []
    try:
        j = json.loads(ps)
        j = j if isinstance(j, list) else [j]
        for p in j:
            cmd = (p.get("CommandLine") or "").replace("\\", "/")
            pid = p.get("ProcessId")
            if not cmd or "cockpit" in cmd:
                continue
            kind = ("训练" if "train.py" in cmd or "train_gui" in cmd else
                    "实验探针" if "/probes/" in cmd or "probes\\" in cmd else "其他")
            procs.append({"pid": pid, "kind": kind,
                          "cmd": cmd.split("/")[-1][:70]})
    except Exception:
        pass
    st["procs"] = procs

    # 所有训练卷（含 status.json → 正在跑的）
    all_runs = []
    for d in glob.glob(os.path.join(ROOT, "runs", "*")):
        if not os.path.isdir(d):
            continue
        tag = os.path.basename(d)
        sfp = os.path.join(d, "status.json")
        running, ep_now, step_now, total_ep = False, None, None, None
        if os.path.isfile(sfp):
            try:
                j = json.load(open(sfp, encoding="utf-8"))
                running = bool(j.get("running"))
                ep_now = j.get("epoch")
                step_now = j.get("step")
                total_ep = j.get("epochs")
            except Exception:
                pass
        hfp = os.path.join(d, "history.csv")
        mt = max([os.path.getmtime(p) for p in (sfp, hfp) if os.path.isfile(p)] or [0])
        all_runs.append({"tag": tag, "running": running, "epoch": ep_now,
                         "step": step_now, "epochs": total_ep, "mtime": mt})
    all_runs.sort(key=lambda x: -x["mtime"])
    st["all_runs"] = all_runs[:12]

    # 最新训练卷
    best_dir, best_mt = None, 0
    for hf in glob.glob(os.path.join(ROOT, "runs", "*", "history.csv")):
        mt = os.path.getmtime(hf)
        if mt > best_mt:
            best_dir, best_mt = hf, mt
    if best_dir:
        tag = os.path.basename(os.path.dirname(best_dir))
        try:
            rows = list(csv.reader(open(best_dir, encoding="utf-8-sig")))
            hdr, data = rows[0], rows[1:]
            last = data[-1] if data else []
            vals = {hdr[i]: last[i] for i in range(min(len(hdr), len(last)))}
            # 若有 status.json，用实时数据覆盖（history 每 epoch 才写一次）
            sfp = os.path.join(ROOT, "runs", tag, "status.json")
            sjs = {}
            if os.path.isfile(sfp):
                try:
                    sjs = json.load(open(sfp, encoding="utf-8"))
                except Exception:
                    sjs = {}
            st["train"] = {
                "tag": tag,
                "epoch": sjs.get("epoch", vals.get("epoch", "?")),
                "step": sjs.get("step"),
                "steps_per_epoch": sjs.get("steps_per_epoch"),
                "n_epochs": sjs.get("epochs", 50),
                "running": bool(sjs.get("running")),
                "loss": sjs.get("loss"),
                "val_loss_now": sjs.get("val_loss"),
                "best_val": sjs.get("best_val"),
                "updated": sjs.get("updated", ""),
                "train_loss": vals.get("train_loss", ""),
                "val_loss": vals.get("val_loss", ""),
                "lr": vals.get("lr", ""),
                "mae_dist": vals.get("mae_dist", ""),
                "val_trend": [float(r[2]) for r in data[-12:] if len(r) > 2],
                "mtime": time.strftime("%H:%M:%S", time.localtime(best_mt)),
            }
        except Exception:
            pass
    st["verdicts"] = VERDICTS
    st["todos"] = TODOS

    # 数据生成进度（任何 status=running 的 data 目录 meta）
    gens = []
    for mp in glob.glob(os.path.join(ROOT, "data", "*", "*", "meta.json")) + \
             glob.glob(os.path.join(ROOT, "data", "*", "meta.json")):
        try:
            m = json.load(open(mp, encoding="utf-8"))
            if m.get("status") == "running" or \
               (m.get("status") == "done" and
                time.time() - os.path.getmtime(mp) < 300):
                gens.append({"dir": mp.replace(ROOT, "").replace("\\", "/"),
                             "actual": m.get("actual", 0),
                             "total": m.get("total", 0),
                             "status": m.get("status"),
                             "rejects": m.get("rejects", 0)})
        except Exception:
            pass
    st["gens"] = gens[-6:]
    return st


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>pyfai 驾驶舱</title>
<style>
 body{background:#10141c;color:#d8dee9;font-family:Consolas,monospace;
      margin:0;padding:18px}
 h1{font-size:19px;color:#7fd0ff;margin:0 0 12px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;max-width:1180px}
 .card{background:#171d29;border:1px solid #263043;border-radius:10px;
       padding:12px 14px}
 .card h2{font-size:13px;color:#88c0d0;margin:0 0 8px;
          border-bottom:1px solid #263043;padding-bottom:6px}
 table{border-collapse:collapse;font-size:12px;width:100%}
 td,th{padding:3px 8px;border-bottom:1px solid #202939;text-align:left}
 th{color:#5e81ac;font-weight:400}
 .ok{color:#a3be8c}.warn{color:#ebcb8b}.bad{color:#bf616a}
 .big{font-size:22px;color:#7fd0ff}
 #clock{float:right;color:#5e81ac;font-size:12px}
 .todo-done{color:#5e81ac;text-decoration:line-through}
</style></head><body>
<h1>🔭 pyfai-project 驾驶舱 <span id="clock"></span></h1>
<div class="grid">
 <div class="card"><h2>系统</h2><div id="gpu" class="big">…</div>
   <table id="procs"></table></div>
 <div class="card"><h2>训练（自动检测最新卷）</h2><div id="train"></div>
   <svg id="spark" width="100%" height="60"></svg></div>
 <div class="card" style="grid-column:1/3"><h2>全部训练卷</h2>
   <div id="runs"></div></div>
 <div class="card" style="grid-column:1/3"><h2>最新战果</h2>
   <div id="verdicts"></div></div>
 <div class="card" style="grid-column:1/3"><h2>数据生成</h2>
   <div id="gens"><i>无生成任务</i></div></div>
 <div class="card" style="grid-column:1/3"><h2>待办</h2>
   <div id="todos"></div></div>
</div>
<script>
async function tick(){
 try{
  const s = await (await fetch('/json')).json();
  document.getElementById('clock').textContent = s.time;
  document.getElementById('gpu').textContent = s.gpu || 'GPU 空闲';
  document.getElementById('procs').innerHTML =
   (s.procs.length? '<tr><th>PID</th><th>类型</th><th>命令</th></tr>'+
    s.procs.map(p=>`<tr><td>${p.pid}</td><td class="warn">${p.kind}</td>
     <td>${p.cmd}</td></tr>`).join('') :
     '<tr><td class="ok">无任务在跑</td></tr>');
  if(s.train){
   const t=s.train;
   const rn=t.running?'<span class="ok">● 运行中</span>':'<span class="bad">○ 已结束</span>';
   const stepTxt=(t.step!=null&&t.steps_per_epoch)?(' step '+t.step+'/'+t.steps_per_epoch):'';
   document.getElementById('train').innerHTML =
    `<b class="big">${t.tag}</b> &nbsp;${rn}&nbsp; epoch <b>${t.epoch}/${t.n_epochs}</b>${stepTxt}
     &nbsp;<span class="warn">${t.updated||t.mtime} 更新</span><br>
     loss <b>${t.loss!=null?Number(t.loss).toFixed(4):'—'}</b> | val ${t.val_loss_now!=null?Number(t.val_loss_now).toFixed(4):t.val_loss} | lr ${t.lr}<br>
     mae_dist ${t.mae_dist} &nbsp; best_val ${t.best_val!=null?Number(t.best_val).toFixed(4):'—'}`;
   const w=460,h=60, v=t.val_trend;
   if(v.length>1){
    const lo=Math.min(...v), hi=Math.max(...v);
    const pts=v.map((y,i)=>`${i/(v.length-1)*w},${h-8-(y-lo)/(hi-lo+1e-9)*(h-16)}`).join(' ');
    const up = v[v.length-1] > v[0];
    document.getElementById('spark').innerHTML =
     `<polyline points="${pts}" fill="none" stroke="${up?'#bf616a':'#a3be8c'}" stroke-width="2"/>`+
     `<text x="2" y="12" fill="#5e81ac" font-size="10">val_loss 近12轮（升=过拟合警讯）</text>`;
   }
  } else document.getElementById('train').innerHTML='<i>无训练记录</i>';
  document.getElementById('runs').innerHTML =
   (s.all_runs&&s.all_runs.length)? '<tr><th>训练卷</th><th>状态</th><th>epoch</th><th>更新</th></tr>'+
    s.all_runs.map(r=>`<tr><td>${r.tag}</td>
     <td class="${r.running?'ok':'warn'}">${r.running?'● 运行中':'○ 结束'}</td>
     <td>${r.epoch!=null?r.epoch:'—'}${r.epochs?'/'+r.epochs:''}</td>
     <td>${r.mtime?new Date(r.mtime*1000).toLocaleTimeString():'—'}</td></tr>`).join('')
    : '<tr><td class="ok">无训练卷</td></tr>';
  document.getElementById('verdicts').innerHTML = s.verdicts.map(v=>
   `<h2 style="color:#7fd0ff">${v.title}</h2><table><tr>${
    v.head.map(h=>`<th>${h}</th>`).join('')}</tr>${
    v.rows.map(r=>`<tr>${r.map((c,i)=>`<td class="${i?'ok':''}">${c}</td>`).join('')}</tr>`).join('')}</table>`).join('<br>');
  document.getElementById('todos').innerHTML = s.todos.map(([d,t])=>
   `<div class="${d==='done'?'todo-done':'warn'}">${d==='done'?'✔':'◻'} ${t}</div>`).join('');
  document.getElementById('gens').innerHTML =
   (s.gens && s.gens.length)? s.gens.map(g=>{
    const pc = g.total? (100*g.actual/g.total).toFixed(1) : 0;
    return `<div style="margin:4px 0">${g.dir}
     <span class="${g.status==='running'?'warn':'ok'}">[${g.status}]</span>
     ${g.actual}/${g.total} (${pc}%) 拒绝${g.rejects}
     <div style="background:#202939;height:6px;border-radius:3px">
      <div style="background:#7fd0ff;width:${pc}%;height:6px;border-radius:3px"></div></div></div>`;
   }).join('') : '<i>无生成任务</i>';
 }catch(e){ document.getElementById('clock').textContent='连接失败，等下一轮'; }
}
tick(); setInterval(tick, 3000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/json":
            b = json.dumps(collect(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        else:
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


if __name__ == "__main__":
    print(f"cockpit on http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
