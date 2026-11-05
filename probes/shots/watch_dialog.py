# -*- coding: utf-8 -*-
"""点击 打开图像 后枚举新出现的顶层窗口。"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pywinauto import Desktop

UIA = Desktop(backend="uia")
win = UIA.window(title="衍射图盲解 · 相鉴定 + 几何标定")
win.set_focus()
time.sleep(0.5)
before = set()
for w in UIA.windows():
    try:
        before.add(w.window_text() + "|" + w.element_info.class_name)
    except Exception:
        pass

btn = win.child_window(control_type="Button", title="打开图像")
btn.click_input()
print("clicked, waiting for new windows...")
for i in range(10):
    time.sleep(0.7)
    for w in UIA.windows():
        try:
            k = w.window_text() + "|" + w.element_info.class_name
            if k not in before and w.is_visible():
                print(f"NEW after {(i+1)*0.7:.1f}s: title={w.window_text()!r} class={w.element_info.class_name!r}")
                # 打印其子控件概览
                for c in w.descendants()[:40]:
                    print("   ", c.element_info.control_type, repr((c.window_text() or "")[:50]))
        except Exception:
            pass
