# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"D:\pyfai-project")
sys.path.insert(0, r"D:\pyfai-project\probes")
os.chdir(r"D:\pyfai-project")
import solver_core as sc
from gui_solver import load_any

img = load_any(r"data\real\MX_ceria\ceria_150_1_0001.cbf")
r = sc.solve_image(img, 172e-6, lam=0.972386)
D = r["sol"][2] * r["s"] * 172e-3
print(f"回退后 ceria_150: D={D:.2f} Δ={abs(D - 149.045):.3f} "
      f"m={r['m']} rms={r['rms']:.2f} ps={r['ps']:.3f}")
print("期望(parab 行):   D=148.97 Δ=0.075 m=9 rms=0.46 ps=0.750")
