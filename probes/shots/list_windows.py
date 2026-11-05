# -*- coding: utf-8 -*-
"""列出当前所有顶层可见窗口。"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pywinauto import Desktop

for w in Desktop(backend="uia").windows():
    try:
        if w.is_visible():
            print(f"title={w.window_text()!r:60s} class={w.element_info.class_name!r}")
    except Exception as e:
        print("err", e)
