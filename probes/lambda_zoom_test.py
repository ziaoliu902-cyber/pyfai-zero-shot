# -*- coding: utf-8 -*-
"""验证用户的想法：真实图以 poni 为中心 ×(λ_t/λ_r) 缩放后喂 NN 能否工作。

若 NN 识别"假想压缩物相"的环系模式：dist_pred 应 ≈ dist_true × s（几何等价）。
对照组：λ 换算律 dist_true = dist_pred × λ_t/λ_r（不缩放直接解释）。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio
import pyFAI
from scipy.ndimage import affine_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")
S = 1.5406 / 0.972386          # 训练λ/真实λ = 1.5843


def to_nn(img, hw=(490, 521)):
    from scipy.ndimage import zoom
    H, W = hw
    img = img[:img.shape[0] - img.shape[0] % 2, :img.shape[1] - img.shape[1] % 2]
    b = img.reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2).mean((1, 3))
    return zoom(b, (H / b.shape[0], W / b.shape[1]), order=1).astype(np.float32)


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

    print(f"缩放因子 s=λ_t/λ_r={S:.4f}；等效目标 = dist_true×s")
    print(f"{'档':>5s}{'dist真值':>9s}{'等效目标':>9s}{'NN(缩放图)':>11s}"
          f"{'Δmm':>8s}{'NN裸(换算)':>11s}{'Δmm':>8s}")
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(
            REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data.astype(np.float32)
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = p.dist
        # 以真值 poni 为中心放大 s 倍（能力验证口径）
        cy, cx = p.poni1 / 0.000172, p.poni2 / 0.000172
        H, W = img.shape
        mat = np.eye(3)
        mat[0, 0] = mat[1, 1] = S                       # 放大
        mat[0, 2] = cy - S * cy
        mat[1, 2] = cx - S * cx
        z = affine_transform(img, mat[:2, :2], offset=mat[:2, 2],
                             output_shape=(H, W), order=1)
        with torch.no_grad():
            x = torch.from_numpy(np.log1p(np.clip(to_nn(z), 0, None))
                                 )[None, None].cuda()
            mu, _ = model(x)
        pred_s = norm_to_param(mu[0].float().cpu().numpy().astype(np.float64),
                               ds.norm_consts)[0]
        # 对照：裸图 NN 预测 × λ 换算
        with torch.no_grad():
            x0 = torch.from_numpy(np.log1p(
                to_nn(fabio.open(os.path.join(
                    REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data))
                )[None, None].cuda()
            mu0, _ = model(x0)
        pred_bare = norm_to_param(mu0[0].float().cpu().numpy().astype(np.float64),
                                  ds.norm_consts)[0] * S
        tgt = gt * S
        print(f"{d:5d}{gt:9.4f}{tgt:9.4f}{pred_s:11.4f}"
              f"{(pred_s-tgt)*1000:8.1f}{pred_bare:11.4f}"
              f"{(pred_bare-gt)*1000:8.1f}")


if __name__ == "__main__":
    main()
