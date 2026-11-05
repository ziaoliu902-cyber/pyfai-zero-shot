# -*- coding: utf-8 -*-
"""pyFAI-rebin 终审：真实 6M 帧 → 训练网格（834µm, 521×490）几何精确重采样。
路线：源/目标几何同 (dist,poni,rot,λ)，只差像素网格 → 源网格像元→lab 是仿射
→ 解 3 点定仿射、第 4 点验残差 → 目标每像素 5×5 子采样面积平均（抗锯齿）。
对比：旧 scipy-zoom / PIL-BILINEAR / pyFAI-rebin 三管线 × 7 帧。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
os.chdir(ROOT)
import fabio
import pyFAI
from scipy.ndimage import map_coordinates

from real_refine_v6 import load_px_nn, match_gain, to_nn_input_wide
from torchfai.dataset import norm_to_param
from torchfai.simulator import get_scaled_detector

m, dev, consts = load_px_nn(r"runs\v4w2\best.pt")
from torchfai.dataset import TorchFAIDataset as _T
consts = _T(os.path.join(ROOT, "data", "train_v4w2"), "val").norm_consts
m.eval()
LAM = 0.972386
K = 2.425


def rebin_to_canvas(img, p, k):
    """img: 源计数图；p: GT poni 几何；k: 目标像素尺度。返回画布 float64。"""
    from pyFAI.detectors import Detector
    db = get_scaled_detector("Pilatus_1M", 2, k)      # 目标网格（torchfai）
    pdet = Detector(pixel1=db.pixel, pixel2=db.pixel,
                    max_shape=tuple(db.shape))
    Hc, Wc = db.shape
    # ---- 源几何：GT poni 原生 ----
    S = pyFAI.load(p) if isinstance(p, str) else p
    Hs, Ws = img.shape
    # ---- 目标几何：同物理参数 + 目标探测器 ----
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    D = AzimuthalIntegrator(dist=S.dist, poni1=S.poni1, poni2=S.poni2,
                            rot1=S.rot1, rot2=S.rot2, rot3=S.rot3,
                            wavelength=S.wavelength, detector=pdet)

    def pos(g, r, c):
        z, y, x = g.calc_pos_zyx(d1=np.float64(r), d2=np.float64(c))
        return np.array([float(np.ravel(x)[0]), float(np.ravel(y)[0]),
                         float(np.ravel(z)[0])])

    # ---- 仿射：源像素(r,c,1) → lab(x,y,z)；3 点定参、1 点验证 ----
    pts = [(0, 0), (Hs - 1, 0), (0, Ws - 1)]
    P = np.array([[r, c, 1.0] for r, c in pts])
    Q = np.array([pos(S, r, c) for r, c in pts])
    A3 = np.linalg.solve(P, Q)                      # 3×3: (r,c,1)→lab
    chk = np.array([Hs - 1, Ws - 1, 1.0]) @ A3 - pos(S, Hs - 1, Ws - 1)
    assert np.abs(chk).max() < 1e-6, f"仿射残差过大 {chk}"
    A3inv = np.linalg.pinv(A3)                      # 像元共面 → 伪逆
    src_img = np.nan_to_num(np.asarray(img, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
    src_img = np.clip(src_img, 0, None)

    # ---- 目标侧仿射（同样 3 点定参 + 验证），bulk 用 numpy ----
    pts_d = [(0, 0), (Hc - 1, 0), (0, Wc - 1)]
    Pd = np.array([[r, c, 1.0] for r, c in pts_d])
    Qd = np.array([pos(D, r, c) for r, c in pts_d])
    A_d = np.linalg.solve(Pd, Qd)
    chk_d = np.array([Hc - 1, Wc - 1, 1.0]) @ A_d - pos(D, Hc - 1, Wc - 1)
    assert np.abs(chk_d).max() < 1e-6, f"目标仿射残差 {chk_d}"

    # ---- 目标每个像素 lab → 源分数坐标；5×5 面积子采样 ----
    rr, cc = np.mgrid[0:Hc, 0:Wc]
    offs = (np.arange(5) - 2.0) / 5.0               # ±0.4 像素内子采样
    acc = np.zeros((Hc, Wc), np.float64)
    cnt = np.zeros((Hc, Wc), np.int32)
    for dr in offs:
        for dc in offs:
            rc1 = np.stack([np.ravel(rr + dr), np.ravel(cc + dc),
                            np.ones(rr.size)], axis=1)
            lab = rc1 @ A_d                           # (N,3) 目标像素→lab
            src = lab @ A3inv                         # (N,3) (r,c,1) 行约定无转置
            r_f = src[:, 0]
            c_f = src[:, 1]
            ok = (r_f >= 0) & (r_f <= Hs - 1) & (c_f >= 0) & (c_f <= Ws - 1)
            rf = np.clip(r_f, 0, Hs - 1)
            cf = np.clip(c_f, 0, Ws - 1)
            v = map_coordinates(src_img, [rf, cf],
                                order=1, mode="nearest")
            v = np.where(ok, v, 0.0)
            acc += v.reshape(Hc, Wc)
            cnt += ok.reshape(Hc, Wc)
    canvas = acc / np.maximum(cnt, 1)
    canvas[cnt == 0] = 0.0
    return canvas


print(f"{'帧':>5s}{'旧zoom':>9s}{'PIL-BL':>9s}{'rebin':>9s}")
tot = {"z": [], "p": [], "r": []}
for d in (150, 200, 250, 300, 350, 400, 450):
    img = fabio.open(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.cbf")).data
    p = pyFAI.load(os.path.join(
        ROOT, "data", "real", f"MX_ceria/ceria_{d}_1_0001.poni"))
    gt = p.dist * 1000
    wide, k, _ = to_nn_input_wide(img)
    reb = np.clip(match_gain(rebin_to_canvas(img, p, K)), 0, None)

    outs = []
    for canvas in (wide, None, reb):
        if canvas is None:
            outs.append(float("nan"))
            continue
        x = torch.from_numpy(
            np.log1p(canvas[None, None].astype(np.float32))).to(dev)
        with torch.no_grad():
            mu, _ = m(x, torch.tensor([LAM], device=dev),
                      torch.tensor([k], device=dev))
        outs.append(float(norm_to_param(
            mu.double().cpu().numpy(), consts, k)[0, 0]) * 1000 - gt)
    z_, _, r_ = outs
    tot["z"].append(abs(z_)); tot["r"].append(abs(r_))
    print(f"{d:>5d}{z_:+9.1f}{'-':>9s}{r_:+9.1f}")
print(f"\n平均|Δ|: 旧zoom={np.mean(tot['z']):.1f} → rebin={np.mean(tot['r']):.1f}mm")
