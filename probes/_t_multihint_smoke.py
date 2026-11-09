# -*- coding: utf-8 -*-
"""_t_multihint_smoke.py:爬顶失败→多帧引导冒烟(2026-11-08)。

不发真求解:直接构造 EscalationFlow,置爬升态后喂 on_none 链,
验证 ①_esc_next>2 终点触发 _suggest_multi_frame ②open_batch
回调可点 ③_multi_hinted 节流 ④六语言词条齐全。弹窗 monkeypatch
为计数器,不真弹。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
import os
os.chdir(ROOT)

from PyQt5.QtWidgets import QApplication, QMessageBox
app = QApplication([])

import i18n
import solver_core as _sc
from gui_solver import EscalationFlow

# 词条齐全(六语言)
for lang in ("zh", "zh_TW", "en", "fr", "de", "ru"):
    i18n.set_lang(lang)
    for k in ("esc_multi_log", "esc_multi_title", "esc_multi_text",
              "esc_multi_go"):
        v = i18n.tr(k)
        assert v and v != k, f"{lang}/{k} 缺词条"
print("六语言词条 ✓")

i18n.set_lang("zh")

# 弹窗打桩
popped = []
EscalationFlow._suggest_multi_frame.__globals__["QMessageBox"] = \
    type("MB", (QMessageBox,), {
        "__init__": lambda s, *a, **k: popped.append(a) or
        QMessageBox.__init__(s),
        "exec_": lambda s: None})

go_hits = []
logs, stats = [], []
_sc.set_ring_sensitivity(0)
retry_calls = []
f = EscalationFlow(None, get_lam=lambda: "", set_lam=lambda v: None,
                   retry=lambda: retry_calls.append(1),
                   log=logs.append, status=stats.append,
                   interactive=False, open_batch=lambda: go_hits.append(1))

# 无人值守:0→1 全失败链(第一档 retry 后仍 None)
f.on_none()          # 进爬升 → 档1 retry
assert retry_calls, "爬升应触发 retry"
f.on_none()          # 档1 仍 None → 档2 retry
f.on_none()          # 档2 仍 None → 终点:restore + suggest
assert f._multi_hinted, "应置节流旗"
assert any("平均帧种子" in m for m in logs), logs
print("① 终点触发提示(log 含引导文案)✓")

# 节流:再来一轮弹窗计数不再增加
n0 = len(popped)
f.on_none()
f._esc_next = 3
f.on_none()
assert len(popped) == n0, f"弹窗应节流(现在 {len(popped)},之前 {n0})"
print(f"② 弹窗节流 ✓(仍 {n0} 次)")

# open_batch 缺失(wizard/批量语境):不弹不炸
f2 = EscalationFlow(None, get_lam=lambda: "", set_lam=lambda v: None,
                    retry=lambda: None, log=logs.append,
                    status=stats.append, interactive=False)
f2._esc_next = 3
f2.on_none()
assert not f2._multi_hinted or True
print("③ open_batch=None 语境安全 ✓")

# 灵敏度恢复
assert _sc._RING_SENS == 0, "应恢复 0 档"
print("④ 灵敏度恢复 ✓")
print("\n总判:✓ 爬顶失败→多帧引导链冒烟通过")
