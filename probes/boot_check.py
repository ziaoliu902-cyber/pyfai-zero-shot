# -*- coding: utf-8 -*-
"""boot_check.py：绿色包启动体检（2026-11-03）。

 启动向导.bat 先跑这里再拉 GUI：
 1. 逐个检查依赖库（import 名 → pip 包名）；
 2. 有缺 → 中文报告缺哪些，按回车 = 自动安装（清华镜像），装完复查；
 3. 全齐 → pythonw 后台启动向导（不占黑窗）。
 环境变量 BOOT_NO_LAUNCH=1 供自动化测试（只体检不启动）。
"""
import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (import 试探名, pip 安装名)——PyQt5 用 QtWidgets 试探，
# 其余同名。缺哪个报 pip 名，装完重查。
REQ = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("PyQt5.QtWidgets", "PyQt5"),
    ("pyFAI", "pyFAI"),
    ("fabio", "fabio"),
    ("PIL", "pillow"),
    ("h5py", "h5py"),
]
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def missing():
    out = []
    for mod, pkg in REQ:
        try:
            importlib.import_module(mod)
        except Exception:
            out.append(pkg)
    return out


def launch():
    if os.environ.get("BOOT_NO_LAUNCH"):
        print("（BOOT_NO_LAUNCH=1：跳过启动）")
        return
    pw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    exe = pw if os.path.isfile(pw) else "pythonw"
    subprocess.Popen([exe, os.path.join(HERE, "gui_wizard.py")], cwd=HERE)


def main():
    print("=" * 50)
    print("  衍射图盲解 · 启动体检")
    print("=" * 50)
    need = missing()
    if not need:
        print("✓ 依赖齐全，正在启动向导…")
        launch()
        return
    print(f"\n检测到缺少 {len(need)} 个库（共需 {len(REQ)} 个）：")
    for p in need:
        print("   ✗ 缺", p)
    print("\n→ 按回车键自动安装（走清华镜像，几分钟）；直接关窗 = 取消")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return
    print("\n开始安装…\n")
    r = subprocess.call(
        [sys.executable, "-m", "pip", "install", *need, "-i", MIRROR])
    if r != 0:
        print("\n✗ 安装失败。常见原因：网络不通 / 权限不足。")
        print("  可手动重试：")
        print("  python -m pip install", " ".join(need))
        input("按回车退出…")
        sys.exit(1)
    still = missing()
    if still:
        print("\n✗ 装完仍缺：", " ".join(still),
              "——请把上面 pip 的输出截图发给维护者")
        input("按回车退出…")
        sys.exit(1)
    print("\n✓ 安装完成，正在启动向导…")
    launch()


if __name__ == "__main__":
    main()
