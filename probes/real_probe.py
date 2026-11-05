# -*- coding: utf-8 -*-
"""真实数据对拍 v0：pyFAI 官方实拍标样图 × base_v3（sim-to-real 裸跑）。

资产（data/real/，silx.org/pub/pyFAI，官方多帧标定 poni 作真值）：
  - Pilatus1M.edf + .poni      λ=1.000Å  探测器同款；实测峰≈0.111=AgBh 一级
  - MX_ceria/ceria_{150..450}.cbf + .poni  λ=0.972Å  CeO₂ 七距离档（Pilatus6M）
  - alumina/al2o3_00xx.edf(51帧) + poni    λ=0.708Å  Al₂O₃（Frelon+spline）
  - LaB6_gonio_D2AM/(15帧) + D5Geom-2018.h5  备用

步骤：物相验证（积分峰 vs 理论 q）→ 重采样 490×521 → base_v3 裸跑 → 对比 poni 真值。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchfai.phases import phase_qi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")

import fabio
import pyFAI
try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
except ImportError:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator


def load_geom(poni_fp):
    """读 poni（兼容旧版 PixelSize/Spline 格式：手工解析）。"""
    try:
        return pyFAI.load(poni_fp)
    except Exception:
        kv = {}
        for ln in open(poni_fp, encoding="utf-8"):
            if ":" in ln:
                k, v = ln.split(":", 1)
                kv[k.strip()] = v.strip()
        return AzimuthalIntegrator(
            dist=float(kv["Distance"]), poni1=float(kv["Poni1"]),
            poni2=float(kv["Poni2"]), rot1=float(kv["Rot1"]),
            rot2=float(kv["Rot2"]), rot3=float(kv["Rot3"]),
            wavelength=float(kv.get("Wavelength", 1e-10)),
            detector="Perkin", pixel1=float(kv["PixelSize1"]),
            pixel2=float(kv["PixelSize2"]))


def check_phase(img, poni_fp, phase):
    ai = AzimuthalIntegrator.sload(poni_fp)
    q, I = ai.integrate1d(img, 2000, unit="q_A^-1", method="bbox")
    m = ~np.isnan(I)
    q, I = q[m], I[m]
    pk = []
    for i in range(2, len(q) - 2):
        if I[i] == max(I[i - 2:i + 3]) and I[i] > 0.15 * I.max() and \
                (not pk or q[i] - pk[-1] > 0.03):
            pk.append(q[i])
    qs = np.array(phase_qi(phase)[0])
    print(f"    实测峰 {len(pk)}: " + " ".join(f"{p:.3f}" for p in pk[:8]))
    print(f"    {phase} 理论: " + " ".join(f"{p:.3f}" for p in qs[:8]))
    if pk:
        print(f"    首峰失配 {abs(qs - pk[0]).min():.4f} A^-1"
              f"（<0.01 = 物相+λ 验证通过）")


def to_nn_input(img, hw=(490, 521)):
    from scipy.ndimage import zoom
    H, W = hw
    img = np.clip(img.astype(np.float64), 0, None)
    img = img[:img.shape[0] - img.shape[0] % 2, :img.shape[1] - img.shape[1] % 2]
    b = img.reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2).mean((1, 3))
    return zoom(b, (H / b.shape[0], W / b.shape[1]), order=1).astype(np.float32)


def to_nn_input_phys(img, hw=(521, 490)):
    """物理一致 NN 输入（2026-10-29 修复版，替代 to_nn_input）。

    语义 = "训练网格对准的视域"：中央裁 2×目标维 → 2×2 均值 bin。
    - 有效像素恒为训练像素（1M 系 344µm），环尺度/环宽与训练分布一致
    - 不缩放（无 zoom 各向异性）、不转置（训练图 (521,490) 竖版）
    - 代价：视域外环丢弃 → 远距信息损失（覆盖洞，数据侧另修）
    to_nn_input 的三个毛病（转置 / 非均匀缩放 / 环尺度破坏）全部去除。
    输入须 ≥ (2H, 2W)；负值/NaN 先清洗（真实图暗流减出负值、坏点）。
    """
    H, W = hw
    img = np.nan_to_num(np.clip(np.asarray(img, np.float64), 0, None))
    if img.shape[0] < 2 * H or img.shape[1] < 2 * W:
        raise ValueError(f"图 {img.shape} 小于 2x 目标 ({2*H},{2*W})，"
                         "物理一致输入不支持（小探测器需零垫方案，另议）")
    r0 = (img.shape[0] - 2 * H) // 2
    c0 = (img.shape[1] - 2 * W) // 2
    crop = img[r0:r0 + 2 * H, c0:c0 + 2 * W]
    return crop.reshape(H, 2, W, 2).mean((1, 3)).astype(np.float32)


def main():
    import torch
    from torchfai.dataset import TorchFAIDataset, norm_to_param
    from torchfai.model import ResNet18Geo

    ds = TorchFAIDataset(os.path.join(ROOT, "data/train_v3"), split="test")
    ck = torch.load(os.path.join(ROOT, "runs/base_v3/last.pt"),
                    map_location="cpu", weights_only=False)
    model = ResNet18Geo()
    model.load_state_dict(ck["model"])
    model.eval().cuda()

    ceria = [(f"MX_ceria/ceria_{d}_1_0001.cbf",
              f"MX_ceria/ceria_{d}_1_0001.poni", "CeO2")
             for d in (150, 200, 250, 300, 350, 400, 450)]
    alum = [(f"alumina/al2o3_{i:04d}.edf",
             "alumina/al2o3_00_max_51_frames.poni", "Al2O3")
            for i in (0, 25, 50)]
    cases = ([("Pilatus1M.edf", "Pilatus1M.poni", "AgBh")] + ceria + alum)

    print(f"{'样本':26s}{'相':5s}{'λÅ':>6s}{'dist真值':>9s}{'dist预测':>9s}"
          f"{'Δmm':>9s}{'poni1Δpx':>9s}{'poni2Δpx':>9s}{'rot1Δmrad':>10s}")
    for n, (img_fp, poni_fp, ph) in enumerate(cases):
        img = fabio.open(os.path.join(REAL, img_fp)).data
        g = load_geom(os.path.join(REAL, poni_fp))
        if n < 3:                                   # 前三个做物相验证
            print(f"  [{img_fp}] shape={img.shape} cnt[{img.min():.0f},"
                  f"{img.max():.0f}]")
            check_phase(img, os.path.join(REAL, poni_fp), ph)
        x = torch.from_numpy(np.log1p(to_nn_input(img)))[None, None].cuda()
        with torch.no_grad():
            mu, lv = model(x)
        pred = norm_to_param(mu[0].float().cpu().numpy().astype(np.float64),
                             ds.norm_consts)
        pxsz = 0.000172
        print(f"{os.path.basename(img_fp):26s}{ph:5s}{g.wavelength*1e10:6.3f}"
              f"{g.dist:9.4f}{pred[0]:9.4f}{(pred[0]-g.dist)*1000:9.1f}"
              f"{(pred[1]-g.poni1)/pxsz:9.1f}{(pred[2]-g.poni2)/pxsz:9.1f}"
              f"{(pred[3]-g.rot1)*1000:10.2f}")


if __name__ == "__main__":
    main()
