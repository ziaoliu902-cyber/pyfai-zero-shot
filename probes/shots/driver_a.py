# -*- coding: utf-8 -*-
"""阶段A：空窗截图 → 打开图像(导航目录→选文件) → 截图2 → 填波长0.9919。"""
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pywinauto import Desktop
from pywinauto.keyboard import send_keys

SHOTS = r"D:\pyfai-project\probes\shots"
UIA = Desktop(backend="uia")


def get_win():
    return UIA.window(title="衍射图盲解 · 相鉴定 + 几何标定")


def shot(win, name):
    win.capture_as_image().save(rf"{SHOTS}\{name}")
    print(f"[SHOT] {name} saved")


def log(*a):
    print(*a, flush=True)


def wait_status(win, prefix, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            txts = [c for c in win.descendants(control_type="Text")]
            for t in txts:
                if t.window_text().startswith(prefix):
                    return t.window_text()
        except Exception:
            pass
        time.sleep(1)
    return None


# ---------- 1. 空窗口 ----------
win = get_win()
assert win.exists(5, 0.5), "主窗口不存在"
win.set_focus()
time.sleep(1.0)
log("窗口标题:", win.window_text(), "rect:", win.rectangle())
shot(win, "shot1_empty.png")

# ---------- 2. 打开图像 ----------
btn_open = win.child_window(control_type="Button", title="打开图像")
btn_open.click_input()
log("已点击 打开图像")

# 等原生文件对话框
dlg = None
t0 = time.time()
while time.time() - t0 < 20:
    try:
        d = UIA.window(title="选择衍射图")
        if d.exists(1, 0.2):
            dlg = d
            break
    except Exception:
        pass
    time.sleep(0.5)
assert dlg is not None, "文件对话框未出现"
dlg.set_focus()
time.sleep(0.6)
log("文件对话框已出现:", dlg.window_text())

# 文件名编辑框：导航到目录
fname_edit = None
for e in dlg.descendants(control_type="Edit"):
    fname_edit = e
    break
assert fname_edit is not None, "未找到文件名输入框"
fname_edit.set_focus()
time.sleep(0.3)
send_keys("^a")
send_keys(r"D:\pyfai-project\data\real\silx_zoo", with_spaces=True, pause=0.01)
send_keys("{ENTER}")
log("已输入目录路径并回车导航")
time.sleep(1.2)

# 验证目录内出现 Pilatus2M.cbf 列表项
items = []
try:
    items = [li.window_text() for li in dlg.descendants(control_type="ListItem")]
except Exception as e:
    log("list item err:", e)
log("对话框列表项(前12):", [i for i in items if "Pilatus" in i or ".cbf" in i][:12])
assert any("Pilatus2M.cbf" in i for i in items), "目录导航失败：未见 Pilatus2M.cbf"

# 输入文件名并点击 打开(O)
fname_edit.set_focus()
time.sleep(0.3)
send_keys("^a")
send_keys("Pilatus2M.cbf", pause=0.01)
time.sleep(0.4)
btn_ok = None
for b in dlg.descendants(control_type="Button"):
    if b.window_text().replace("&", "").startswith("打开"):
        btn_ok = b
        break
if btn_ok is not None:
    btn_ok.click_input()
    log("已点击 打开(O) 按钮")
else:
    send_keys("{ENTER}")
    log("未找到打开按钮，改用回车确认")

# 等加载完成
st = wait_status(win, "已加载", timeout=60)
assert st is not None, "图像加载超时（状态栏未出现 已加载）"
log("状态栏:", st)
time.sleep(1.5)  # 等画布绘制

# ---------- 3. 截图2 ----------
win.set_focus()
time.sleep(0.5)
shot(win, "shot2_loaded.png")

# 右栏文本
panel = None
edits = [e for e in win.descendants(control_type="Edit")]
for e in edits:
    if "尺寸" in (e.window_text() or ""):
        panel = e
        break
if panel is not None:
    log("---右栏加载信息---")
    log(panel.window_text())
else:
    log("右栏 Edit 未匹配到，所有Edit文本:", [e.window_text()[:40] for e in edits])

# ---------- 4. 像素=172 确认，波长=0.9919 ----------
px_edit = None
for e in win.descendants(control_type="Edit"):
    if (e.window_text() or "").strip() == "172":
        px_edit = e
        break
log("像素µm 输入框当前值:", px_edit.window_text() if px_edit else "未找到")

lam_edit = None
for e in win.descendants(control_type="Edit"):
    t = (e.window_text() or "").strip()
    if t == "" and e is not px_edit and "尺寸" not in t:
        # 波长框是顶栏唯一空 Edit（右栏有内容）
        r = e.rectangle()
        wr = win.rectangle()
        if r.top - wr.top < 80:  # 顶栏区域
            lam_edit = e
            break
assert lam_edit is not None, "未找到波长输入框"
lam_edit.set_focus()
time.sleep(0.2)
send_keys("^a")
send_keys("0.9919", pause=0.02)
time.sleep(0.4)
log("波长Å 输入框填入后读回:", repr(lam_edit.window_text()))
assert lam_edit.window_text().strip() == "0.9919", "波长填写失败"

run_btn = win.child_window(control_type="Button", title="▶ 运行求解")
log("▶ 运行求解 按钮可用:", run_btn.is_enabled())
log("PHASE-A OK")
