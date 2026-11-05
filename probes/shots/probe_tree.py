# -*- coding: utf-8 -*-
"""探查 GUI 的 UIA 控件树：确认按钮/文本是否可读、窗口 rect。"""
import time
from pywinauto import Application, Desktop

t0 = time.time()
app = None
win = None
while time.time() - t0 < 60:
    try:
        dlg = Desktop(backend="uia").window(title="衍射图盲解 · 相鉴定 + 几何标定")
        if dlg.exists(1, 0.2):
            win = dlg
            break
    except Exception as e:
        print("retry:", e)
    time.sleep(1)

if win is None:
    print("WINDOW NOT FOUND")
    raise SystemExit(1)

print("window found, rect =", win.rectangle())
try:
    win.set_focus()
    time.sleep(0.5)
    print("focus ok")
except Exception as e:
    print("focus fail:", e)

# 打印前 120 个控件（类型/名称/rect/是否可用）
cnt = 0
for c in win.descendants():
    try:
        name = (c.window_text() or "")[:60]
        typ = c.element_info.control_type
        rid = c.element_info.runtime_id
        auto = c.element_info.automation_id or ""
        print(f"[{cnt:03d}] {typ:14s} name={name!r} auto_id={auto!r} rt={rid} enabled={c.is_enabled()}")
        cnt += 1
    except Exception as e:
        print("  err:", e)
    if cnt > 150:
        break
print("total scanned:", cnt)
