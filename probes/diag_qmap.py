# -*- coding: utf-8 -*-
"""诊断 q-map 残差来源：逐像素对比 t 向量与 q。"""
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\pyfai-project")
from torchfai.refine import DifferentiableRefiner
from pyFAI.geometry import Geometry

ref = DifferentiableRefiner("AgBh", r"D:\pyfai-project\data\train_v2")
db, cfg = ref.db, ref.cfg
g = dict(dist=0.1, poni1=0.5 * db.extent[0], poni2=0.5 * db.extent[1],
         rot1=0.0, rot2=0.0, rot3=0.0)
pyf = Geometry(detector=db.pyfai_det, dist=g["dist"], poni1=g["poni1"],
               poni2=g["poni2"], rot1=0, rot2=0, rot3=0,
               wavelength=cfg["wavelength"])
q_ref = pyf.array_from_unit(shape=db.shape, typ="center", unit="q_A^-1", scale=True)
pn = ref.phys_to_pn(np.array([g["dist"], g["poni1"], g["poni2"], 0, 0, 0]),
                    ref.consts)
with torch.no_grad():
    q_t = ref.q_map(pn.to(ref.device)).cpu().numpy()

d = np.abs(q_t - q_ref)
i, j = np.unravel_index(d.argmax(), d.shape)
print(f"argmax (row={i}, col={j})  q_torch={q_t[i, j]:.6f}  q_pyfai={q_ref[i, j]:.6f}  d={d[i, j]:.2e}")
print(f"q_pyfai 全图范围: {q_ref.min():.3f} .. {q_ref.max():.3f}")

p1v, p2v, p3v = pyf.calc_pos_zyx(d1=np.array([i]), d2=np.array([j]), use_cython=False)
print(f"pyFAI t=(t1,t2,t3)=({p1v[0]:.6f}, {p2v[0]:.6f}, {p3v[0]:.6f}) m")

p1 = i * db.pixel - g["poni1"]
p2 = j * db.pixel - g["poni2"]
two = np.arctan2(np.hypot(p1, p2), g["dist"])
q_mine = 4 * np.pi / (cfg["wavelength"] * 1e10) * np.sin(two / 2)
print(f"我的 p=({p1:.6f}, {p2:.6f}, {g['dist']:.6f})  → q={q_mine:.6f}")
print(f"pyFAI |t_perp|={np.hypot(p1v[0], p2v[0]):.6f}  t3={p3v[0]:.6f}")
print(f"我的  |t_perp|={np.hypot(p1, p2):.6f}  t3={g['dist']:.6f}")
# 反推 pyFAI 认为的 2θ
lam_A = cfg["wavelength"] * 1e10
two_pyf = 2 * np.arcsin(np.clip(q_ref[i, j] * lam_A / (4 * np.pi), 0, 1))
print(f"pyFAI 隐含 2θ={np.degrees(two_pyf):.4f}°  我的 2θ={np.degrees(two):.4f}°  "
      f"pyFAI 几何 2θ={np.degrees(np.arctan2(np.hypot(p1v[0], p2v[0]), p3v[0])):.4f}°")
