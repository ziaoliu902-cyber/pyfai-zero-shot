# -*- coding: utf-8 -*-
"""build_exe.py:傻瓜版打包(2026-11-08)。

产出 dist/<NAME>/(onedir 文件夹版,双击 exe 即用)+ 自动压 zip。
默认打**向导流程版**(gui_wizard.py,五步 QWizard,旧名 DiffBlindWizard);
WIZARD=0 环境变量打主窗口专家版(gui_solver.py,PyFAI-Blind)。
自检:exe --selftest。复用:python probes\\build_exe.py [wizard|expert]

v2 教训:--collect-submodules pyFAI 会拖 silx.gui→torch→onnxruntime
全家(隔离进程崩,exe 2GB)。改为精准 hidden-import + excludes。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

WIZARD = os.environ.get("WIZARD", "1") != "0" or "wizard" in sys.argv
NAME = "DiffBlindWizard" if WIZARD else "PyFAI-Blind"
ENTRY = os.path.join(ROOT, "probes",
                     "gui_wizard.py" if WIZARD else "gui_solver.py")

EXCLUDES = [
    # 深度学习/科学全家桶——本项目零使用(torchfai 名字唬人,实为纯 numpy)
    "torch", "torchgen", "functorch", "onnxruntime", "sympy", "mpmath",
    "numba", "llvmlite", "pandas", "pyarrow", "dask",
    # notebook 生态(matplotlib 可选后端拖进来的)
    "IPython", "jupyter", "jupyter_client", "jupyter_core", "ipykernel",
    "debugpy", "zmq", "tornado", "gevent", "greenlet", "traitlets",
    "prompt_toolkit", "jedi", "parso", "pygments", "rich",
    # GPU/OpenCL(标定纯 CPU)
    "pyopencl", "hdf5plugin", "pyFAI.opencl", "pyFAI.gui", "pyFAI.test",
    "pyFAI.benchmark", "pyFAI.app",
    # silx:pyFAI.io.image 依赖 silx.io——只排 GUI/OpenCL 重面
    "silx.gui", "silx.test", "silx.opencl",
    # 杂项大件
    "tkinter", "test", "pyreadline3", "curses", "lxml", "bs4",
    "requests", "urllib3", "cryptography", "google", "pyasn1",
    "openpyxl", "psutil", "win32com", "pywin",
]   # 注:numexpr 必须收(pyFAI.parallax 依赖)

CMD = [sys.executable, "-m", "PyInstaller",
       "--noconfirm", "--onedir", "--windowed",
       "--name", NAME,
       "--paths", ROOT,
       "--paths", os.path.join(ROOT, "probes"),
       "--hidden-import", "torchfai",
       "--hidden-import", "torchfai.phases",
       "--hidden-import", "h5py",
       "--collect-submodules", "fabio",
       # pyFAI 数据文件(sensors 的 material 定义等;漏收则 Detector
       # 家族 import 即 FileNotFoundError,selftest 实锤)
       "--collect-data", "pyFAI",
       # pyFAI 精准面(geometry/detectors/spline/integrator/poni io)
       "--hidden-import", "pyFAI.geometry",
       "--hidden-import", "pyFAI.detectors",
       "--hidden-import", "pyFAI.spline",
       "--hidden-import", "pyFAI.integrator.azimuthal",
       "--hidden-import", "pyFAI.io",
       "--hidden-import", "silx.io",
       "--hidden-import", "silx.resources",
       "--hidden-import", "silx.utils",
       # pyFAI C 引擎(Cython 二进制;integrator.load_engines 动态取)
       "--collect-submodules", "pyFAI.ext",
       "--collect-submodules", "pyFAI.engines",
       ]
for e in EXCLUDES:
    CMD += ["--exclude-module", e]
CMD.append(ENTRY)

print(" ".join(CMD), flush=True)
r = subprocess.run(CMD)
sys.exit(r.returncode)
