# -*- coding: utf-8 -*-
"""一次性探针：打印 base_v3lamp / base_v3lam 检查点内 opts 与关键键，验收尽职核查。"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for tag in ("base_v3lamp", "base_v3lam"):
    p = os.path.join(ROOT, "runs", tag, "best.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print(f"===== {tag}/best.pt =====")
    print("keys:", sorted(ck.keys()))
    print("epoch:", ck.get("epoch"), " best_val:", ck.get("best_val"))
    print("opts:", ck.get("opts"))
    sd = ck["model"]
    has_lam = any("lam" in k for k in sd)
    print("lam_tent_keys:", [k for k in sd if "lam" in k][:4])
    # λ 触须零初始化是热启动等价的关键；看 lamp 微调后是否已学出非零
    for k in sd:
        if "lam" in k and "weight" in k:
            w = sd[k]
            print(f"  {k}: |w|max={w.abs().max().item():.3e}")
    print()
