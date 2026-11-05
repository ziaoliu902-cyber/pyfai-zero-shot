# -*- coding: utf-8 -*-
"""几何回归网络：手写 ResNet-18 双头（6 均值 + 6 log 方差），零 torchvision 依赖。

输入 (B,1,521,490) float32（log1p 图像）→ 内部重采样 512×512。
输出 mu (B,6)（归一化几何参数，目标域 [-1,1]）与 logvar (B,6)（异方差 NLL 用）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + (x if self.sc is None else self.sc(x))
        return F.relu(out, inplace=True)


class ResNet18Geo(nn.Module):
    """ResNet-18 回归骨干 + 双头。参数量 ~11.2M。

    lam_cond=True：λ 触须——每头一条 Linear(1,6) 加性分支（零初始化）。
    归一化域里 log-dist 的 λ 修正是平移（dist∝λ⁻¹），加性即物理对齐；
    零初始化保证加载 base_v3 权重后初始输出与原模型逐位一致（热启动安全）。
    forward(x, lam)：lam 为 (B,) 波长（Å）。

    px_cond=True：像素尺度触须（10-29 视域域扩展）——同款零初始化加性分支。
    px 为 (B,) 无量纲 k（像素/基准像素；k=1 即旧 1M-bin2 视域）。归一化域里
    环半径 r_px ∝ dist/(k·px₀)，log-dist 的 k 修正是平移（dist∝k），加性对齐。
    forward(x, lam, px)。
    """

    def __init__(self, in_ch: int = 1, out_dim: int = 6, lam_cond: bool = False,
                 px_cond: bool = False, sig_head: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        self.stage1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.stage2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.stage3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.stage4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_mu = nn.Linear(512, out_dim)
        self.head_logvar = nn.Linear(512, out_dim)
        self.out_dim = out_dim
        self.lam_cond = lam_cond
        self.px_cond = px_cond
        if lam_cond:
            self.lam_proj_mu = nn.Linear(1, out_dim)
            self.lam_proj_logvar = nn.Linear(1, out_dim)
            nn.init.zeros_(self.lam_proj_mu.weight)
            nn.init.zeros_(self.lam_proj_mu.bias)
            nn.init.zeros_(self.lam_proj_logvar.weight)
            nn.init.zeros_(self.lam_proj_logvar.bias)
        if px_cond:
            self.px_proj_mu = nn.Linear(1, out_dim)
            self.px_proj_logvar = nn.Linear(1, out_dim)
            nn.init.zeros_(self.px_proj_mu.weight)
            nn.init.zeros_(self.px_proj_mu.bias)
            nn.init.zeros_(self.px_proj_logvar.weight)
            nn.init.zeros_(self.px_proj_logvar.bias)
            # 物理解析初始化（10-29 px_v1 事故修复）：同一张图像素坐标下
            # 环半径不变 ⇒ dist∝k，归一化域里是纯平移 2·log10(k)/(dhi−dlo)。
            # k∈[1,3]、dist∈[0.1,3] ⇒ 平均斜率 = log10(3)/log10(30) ≈ 0.323。
            # 只喂 dist 通道；零初始化学太慢（8 epoch 只学到 1/4，远距全废）。
            with torch.no_grad():
                self.px_proj_mu.weight[0, 0] = float(np.log10(3.0) / np.log10(30.0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head_mu.weight); nn.init.zeros_(self.head_mu.bias)
        nn.init.zeros_(self.head_logvar.weight); nn.init.zeros_(self.head_logvar.bias)
        # σq 辅助头（v4s1 多任务解耦）：预测"图像实测等效环宽"的归一化
        # log10 值（aux 第 8 列）。零初始化 → 从旧权重热启动时初始预测=0
        # （域中点），训练前向逐位兼容；预测经 self.out_sig 暴露，forward
        # 仍返回 (mu, logvar) 二元组——旧推理代码零改动。
        self.sig_head = nn.Linear(512, 1) if sig_head else None
        if self.sig_head is not None:
            nn.init.zeros_(self.sig_head.weight)
            nn.init.zeros_(self.sig_head.bias)

    @staticmethod
    def _lam_norm(lam: torch.Tensor) -> torch.Tensor:
        """波长归一化：Å → [-1,1]，覆盖 [0.5, 2.3]Å。"""
        return ((lam.float().view(-1, 1) - 1.4) / 0.9).clamp(-1.0, 1.0)

    @staticmethod
    def _px_norm(px: torch.Tensor) -> torch.Tensor:
        """像素尺度归一化：k → [-1,1]，覆盖 [1, 3]（6M 宽视域 k≈2.43）。"""
        return ((px.float().view(-1, 1) - 2.0) / 1.0).clamp(-1.0, 1.0)

    def forward(self, x: torch.Tensor, lam: torch.Tensor | None = None,
                px: torch.Tensor | None = None):
        if x.shape[-2:] != (512, 512):                     # (B,1,521,490)→512²
            x = F.interpolate(x.float(), size=(512, 512), mode="bilinear",
                              align_corners=False)
        if x.dim() == 3:                                   # 单样本 (1,H,W)→(1,1,H,W)
            x = x.unsqueeze(1)
        b = x.shape[0]
        f = self.stem(x)
        f = self.stage4(self.stage3(self.stage2(self.stage1(f))))
        f = self.gap(f).view(b, -1)
        mu = self.head_mu(f)
        logvar = self.head_logvar(f).clamp(-6.0, 4.0)      # σ ∈ [0.22, 7.4]，防爆炸/塌缩
        if self.lam_cond and lam is not None:
            lam_n = self._lam_norm(lam)
            mu = mu + self.lam_proj_mu(lam_n)
            logvar = (logvar + self.lam_proj_logvar(lam_n)).clamp(-6.0, 4.0)
        if self.px_cond and px is not None:
            px_n = self._px_norm(px)
            mu = mu + self.px_proj_mu(px_n)
            logvar = (logvar + self.px_proj_logvar(px_n)).clamp(-6.0, 4.0)
        if self.sig_head is not None:
            self.out_sig = self.sig_head(f)         # (B,1) 归一化 log10 环宽
        return mu, logvar
