# -*- coding: utf-8 -*-
"""修复候选验证：物理一致 NN 输入（不缩放、不转置）直接上真实图。

resample_diag 的结论：to_nn_input 的 zoom 缩放破坏环的物理尺度（B/C/E 全崩），
而 D（裁剪+2×2 bin，环尺度与训练一致）在近距工作正常。本探针把 D 语义
搬 to 真实图上——零重训，直接看粗解恢复多少：
  ceria@6M  : 中央裁 1042×980（=训练视域）→ 2×2 bin → (521,490) → NN
  AgBh@1M   : 全图 (1043,981) → 2×2 bin → (521,490) → NN（本就同款探测器）
对照组：昨夜 to_nn_input 老管线（real_refine_v4 CSV 里的粗解误差）。
"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fabio
import pyFAI
import torch

from real_probe import load_geom
from real_refine_v4 import load_nn, nn_coarse
from torchfai.dataset import norm_to_param

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")


def bin2(img):
    img = np.nan_to_num(np.clip(np.asarray(img, np.float64), 0, None))
    h, w = img.shape
    img = img[:h - h % 2, :w - w % 2]
    return img.reshape(h // 2, 2, w // 2, 2).mean((1, 3)).astype(np.float32)


@torch.no_grad()
def pred(m, dev, consts, x, lam):
    t = torch.from_numpy(np.log1p(x))
    t = t.unsqueeze(0).unsqueeze(0).to(dev)
    mu, lv = m(t, torch.tensor([lam], device=dev))
    return norm_to_param(mu.float().cpu().numpy(), consts)[0]


def main():
    models = {t: load_nn(f"runs/{t}/best.pt") for t in ("base_v3lamp", "base_v3lam")}

    print("== ceria@Pilatus6M：中央裁剪训练视域 + 2x2 bin ==")
    for d in (150, 200, 250, 300, 350, 400, 450):
        img = fabio.open(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.cbf")).data
        p = pyFAI.load(os.path.join(REAL, f"MX_ceria/ceria_{d}_1_0001.poni"))
        gt = p.dist
        H, W = img.shape
        r0, c0 = H // 2 - 521 * 2 // 2, W // 2 - 490 * 2 // 2
        crop = img[r0:r0 + 1042, c0:c0 + 980]
        x = bin2(crop)
        line = f"[ceria {d}mm GT={gt*1000:.0f}] "
        for mt, (m, dev, consts) in models.items():
            pp = pred(m, dev, consts, x, 0.972386)
            line += f"{mt}={pp[0]*1000:6.1f}mm({(pp[0]-gt)*1000:+6.1f})  "
        print(line, flush=True)

    print("\n== AgBh@Pilatus1M：全图 2x2 bin（无缩放无转置） ==")
    img = fabio.open(os.path.join(REAL, "Pilatus1M.edf")).data
    p = load_geom(os.path.join(REAL, "Pilatus1M.poni"))
    gt = p.dist
    x = bin2(img)
    line = f"[AgBh@1M GT={gt*1000:.0f}mm] shape={x.shape}  "
    for mt, (m, dev, consts) in models.items():
        pp = pred(m, dev, consts, x, 1.0)
        line += f"{mt}={pp[0]*1000:7.1f}mm({(pp[0]-gt)*1000:+7.1f})  "
    print(line)
    print(f"\n对照组（昨夜 to_nn_input 老管线粗解误差）："
          f"ceria150 +7/+27, ceria300 +354/+302, AgBh -699/-890 (mm)")


if __name__ == "__main__":
    main()
