# -*- coding: utf-8 -*-
"""可微分积分器：NN 粗解 → 泊松似然物理精修。

原理
----
把"正向仿真"重写为可对几何参数自动求导的 torch 计算图：

  1. 像素 → q：pyFAI 旋转矩阵原式（R = R3·R2·R1，PONI 偏移，L=dist）
     → 散射角 2θ = atan2(√(t1²+t2²), t3) → q = 4π/λ·sin(θ)
     （已与 pyFAI 逐像素对拍至 1e-6 Å⁻¹，见 probes/verify_qmap.py）
  2. 图像模型（与 torchfai.simulator 同构）：
        μ(q) = bg0 + bg1·q + Σᵢ Aᵢ·sᵢ²/((q-qᵢ)²+sᵢ²) + 驼峰
     环位 qᵢ 由标准样品 d-spacings 决定（已知物理），几何决定 q-map。
  3. 数据统计为光子计数 → 泊松负对数似然：
        L = Σ_mask (μ - y·log μ)
  4. nuisance 参数按生成器真实结构收紧（关键防伪谷设计）：
        Aᵢ = 标度 · Iᵢ(已知) · exp(-B·qᵢ²) · exp(jᵢ)
     B ∈ DW 真值区间，jᵢ ∈ 抖动真值区间 —— 幅度自由度从 ~15 降到 3。
  5. σ-信任域：几何仅在 NN 自报不确定度的 ±k·σ 内移动，
     防止少环样本滑入竞争性局部谷（不确定性头 × 精修器的组合卖点）。
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .simulator import get_detector
from .phases import phase_qi


def _norm_consts_from_meta(cfg: dict) -> dict:
    """与 torchfai.dataset._norm_consts 相同（独立实现避免循环导入）。"""
    db = get_detector(cfg["detector_name"], cfg["binning"])
    Hm, Wm = db.extent
    return dict(
        dlo=float(np.log10(cfg["dist_range"][0])),
        dhi=float(np.log10(cfg["dist_range"][1])),
        Hm=float(Hm), Wm=float(Wm),
        r_lo=np.array([cfg[f"rot{i}_range"][0] for i in (1, 2, 3)], np.float64),
        r_hi=np.array([cfg[f"rot{i}_range"][1] for i in (1, 2, 3)], np.float64),
    )


class DifferentiableRefiner:
    """单样品几何精修器。用法：

        ref = DifferentiableRefiner("AgBh", "data/train_v2")
        out = ref.refine(img_u16, params_phys_from_nn, sigma_norm=sig_nn)
    """

    def __init__(self, phase: str, data_root: str = "data/train_v2",
                 device: str | None = None, *, detector=None,
                 wavelength_m: float | None = None,
                 dist_range=None):
        """detector/wavelength_m/dist_range：真实数据对拍 override（sim-to-real）。
        传 pyFAI 探测器名经 simulator.get_detector 包一层，或直接给 DetectorBundle。"""
        with open(os.path.join(data_root, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.cfg = self.meta["config"]
        if detector is None:
            self.db = get_detector(self.cfg["detector_name"], self.cfg["binning"])
        elif isinstance(detector, str):
            self.db = get_detector(detector, 1)
        else:
            self.db = detector
        lam = self.cfg["wavelength"] if wavelength_m is None else wavelength_m
        self.lam_A = lam * 1e10
        Hm, Wm = self.db.extent
        c = dict(
            dlo=float(np.log10(dist_range[0] if dist_range else self.cfg["dist_range"][0])),
            dhi=float(np.log10(dist_range[1] if dist_range else self.cfg["dist_range"][1])),
            Hm=float(Hm), Wm=float(Wm),
            r_lo=np.array([self.cfg[f"rot{i}_range"][0] for i in (1, 2, 3)], np.float64),
            r_hi=np.array([self.cfg[f"rot{i}_range"][1] for i in (1, 2, 3)], np.float64),
        )
        self.consts = c
        self.qs, self.Is = phase_qi(phase)            # (n_ring,) 物理常数
        self.n_ring = len(self.qs)
        # 生成器真值区间（防伪谷的先验收紧）
        self.dw_lo, self.dw_hi = self.cfg.get("dw_b_range", [0.0, 0.02])
        jit = self.cfg.get("amp_jitter", [0.5, 1.6])
        self.jit_lo, self.jit_hi = float(np.log(jit[0])), float(np.log(jit[1]))
        self.hq = self.cfg.get("hump_q_range", [0.8, 2.2])
        self.hs = self.cfg.get("hump_sigma_range", [0.15, 0.4])

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 像素网格（pyFAI 像素中心约定 +0.5）
        yy = torch.from_numpy((self.db.yy + 0.5).astype(np.float32)) * float(self.db.pixel)
        xx = torch.from_numpy((self.db.xx + 0.5).astype(np.float32)) * float(self.db.pixel)
        self.yy, self.xx = yy.to(self.device), xx.to(self.device)
        self.dead = torch.from_numpy(self.db.dead).to(self.device)
        self.qs_t = torch.tensor(self.qs, dtype=torch.float32, device=self.device)
        self.Is_t = torch.tensor(self.Is, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------ 几何→q-map
    def q_map(self, pn: torch.Tensor) -> torch.Tensor:
        """归一化几何 (6,) → 探测器 q-map (H,W)。全流程可导。"""
        c = self.consts
        dist = 10.0 ** ((pn[0] + 1.0) / 2.0 * (c["dhi"] - c["dlo"]) + c["dlo"])
        poni1 = (pn[1] + 1.0) / 2.0 * c["Hm"]
        poni2 = (pn[2] + 1.0) / 2.0 * c["Wm"]
        rot1 = (pn[3] + 1.0) / 2.0 * (c["r_hi"][0] - c["r_lo"][0]) + c["r_lo"][0]
        rot2 = (pn[4] + 1.0) / 2.0 * (c["r_hi"][1] - c["r_lo"][1]) + c["r_lo"][1]
        rot3 = (pn[5] + 1.0) / 2.0 * (c["r_hi"][2] - c["r_lo"][2]) + c["r_lo"][2]

        s1, c1 = torch.sin(rot1), torch.cos(rot1)
        s2, c2 = torch.sin(rot2), torch.cos(rot2)
        s3, c3 = torch.sin(rot3), torch.cos(rot3)
        # R = R3 @ R2 @ R1（pyFAI 原式，逐元素精确展开）
        r00, r01, r02 = c3 * c2, c3 * s2 * s1 - s3 * c1, -c3 * s2 * c1 - s3 * s1
        r10, r11, r12 = s3 * c2, s3 * s2 * s1 + c3 * c1, -s3 * s2 * c1 + c3 * s1
        r20, r21, r22 = s2, -c2 * s1, c2 * c1

        p1 = self.yy - poni1          # (H,W)
        p2 = self.xx - poni2
        p3 = dist
        t1 = r00 * p1 + r01 * p2 + r02 * p3
        t2 = r10 * p1 + r11 * p2 + r12 * p3
        t3 = r20 * p1 + r21 * p2 + r22 * p3

        two_theta = torch.atan2(torch.sqrt(t1 * t1 + t2 * t2 + 1e-18), t3)
        q = (4.0 * np.pi / self.lam_A) * torch.sin(two_theta / 2.0)
        return q

    def pn_to_phys(self, pn: torch.Tensor) -> np.ndarray:
        """归一化 → 物理参数（numpy）。"""
        c = self.consts
        with torch.no_grad():
            dist = 10.0 ** ((float(pn[0]) + 1.0) / 2.0 * (c["dhi"] - c["dlo"]) + c["dlo"])
            p1 = (float(pn[1]) + 1.0) / 2.0 * c["Hm"]
            p2 = (float(pn[2]) + 1.0) / 2.0 * c["Wm"]
            r = (torch.tensor([float(pn[3]), float(pn[4]), float(pn[5])]) + 1.0) / 2.0 \
                * torch.tensor(c["r_hi"] - c["r_lo"]) + torch.tensor(c["r_lo"])
        return np.array([dist, p1, p2, float(r[0]), float(r[1]), float(r[2])], np.float64)

    @staticmethod
    def phys_to_pn(p: np.ndarray, c: dict) -> torch.Tensor:
        """物理参数 → 归一化（与 dataset.param_to_norm 一致）。"""
        dist_n = (np.log10(p[0]) - c["dlo"]) / (c["dhi"] - c["dlo"]) * 2.0 - 1.0
        p1_n = p[1] / c["Hm"] * 2.0 - 1.0
        p2_n = p[2] / c["Wm"] * 2.0 - 1.0
        r = 2.0 * (p[3:6] - c["r_lo"]) / (c["r_hi"] - c["r_lo"]) - 1.0
        return torch.tensor([dist_n, p1_n, p2_n, r[0], r[1], r[2]], dtype=torch.float32)

    # ------------------------------------------------------------ 内部渲染
    def _render(self, q, sc, Bj, jit, logs, logbg0, logbg1, logah, qh, logsh):
        s = torch.exp(logs)
        A = torch.exp(sc + jit) * self.Is_t * torch.exp(-Bj * self.qs_t * self.qs_t)
        mu = torch.exp(logbg0) + torch.exp(logbg1) * q
        dq = q.unsqueeze(-1) - self.qs_t                      # (H,W,n_ring)
        mu = mu + (A * s * s / (dq * dq + s * s)).sum(-1)
        mu = mu + torch.exp(logah) * torch.exp(-0.5 * ((q - qh) / torch.exp(logsh)) ** 2)
        return mu.clamp(min=1e-3)

    # ------------------------------------------------------------ 精修主循环
    def refine(self, img_u16: np.ndarray, params_phys_init: np.ndarray,
               iters: int = 300, warmup: int = 80, lr_geo: float = 0.01,
               lr_nui: float = 0.08, include_sat: bool = False,
               sigma_norm=None, trust_k: float = 1.0,
               lr_mult=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
               sigma_range=(0.0015, 0.02), verbose: bool = False,
               soft_prior: float = 0.0, sigma_prior=None,
               trust_floor=None) -> dict:
        """泊松似然精修。两阶段：先冻结几何暖机 nuisance，再联合优化。

        sigma_norm: NN 自报 σ（归一化空间 (6,)）→ σ-信任域 ±trust_k·σ。
        lr_mult: 六参数独立学习率倍率（poni/tilt 常需更大倍率细磨）。
        sigma_range: 环宽 σ 的夹紧范围（收窄到生成器真值区间可破"展宽-
                     倾角"简并：倾向用假展宽冒充椭圆化的环）。
        soft_prior: 软高斯 MAP 先验权重（>0 且给 sigma_norm 时生效）。
                     联合阶段 loss 额外加 w·mean(((pn-pn0)/σ)²)。似然陡
                     方向（dist）罚项可忽略、似然零空间方向（poni↔rot↔
                     dist 联动简并）由 NN 先验定夺。硬笼仍作保险丝。
        返回 dict 含 loss_init（NN 几何处拟合分）与 loss（最优几何处）。
        """
        t0 = time.time()
        dev = self.device
        y = torch.from_numpy(img_u16.astype(np.float32)).to(dev)
        sat = float(self.cfg["sat_clip"])

        with torch.no_grad():
            mask = (~self.dead) & (y > 0)
            if not include_sat:
                mask &= (y < sat)
            n_pix = int(mask.sum())
            imax = float(y.max())
            imed = float(y[mask].median())

        # ---- 优化变量（几何拆成 6 个独立标量：可分参数组独立学习率）----
        pn_init = self.phys_to_pn(np.asarray(params_phys_init, np.float64),
                                  self.consts)
        pns = [pn_init[i].clone().to(dev).float().requires_grad_(True)
               for i in range(6)]
        with torch.no_grad():
            pn0 = pn_init.detach().clone().to(dev).float()
        if sigma_norm is not None:                      # σ-信任域（可每参数独立）
            tk = np.array(trust_k, dtype=float).ravel()
            if tk.size == 1:
                tk = np.full(6, float(tk[0]))
            sig = torch.as_tensor(np.asarray(sigma_norm, np.float64),
                                  dtype=torch.float32, device=dev)
            margin = torch.as_tensor(tk, dtype=torch.float32, device=dev) * sig + 0.01
            # 物理半径下限：σ 塌缩时信任域退化成固定盒子，真值可能落在盒外
            # （实测 rot2 覆盖率仅 39.8%、AgBh 档仅 32%）。
            # trust_floor: (6,) 归一化空间的最小半径，逐通道取 max。
            if trust_floor is not None:
                fl = torch.as_tensor(np.asarray(trust_floor, np.float64),
                                     dtype=torch.float32, device=dev)
                margin = torch.maximum(margin, fl)
            pn_lo = torch.clamp(pn0 - margin, -0.999, 0.999)
            pn_hi = torch.clamp(pn0 + margin, -0.999, 0.999)
        else:
            pn_lo = torch.full((6,), -0.999, device=dev)
            pn_hi = torch.full((6,), 0.999, device=dev)
        # 软先验的 σ 与信任域解耦：信任域 = 搜索范围（要宽，笼子病样本才走得动）；
        # 先验 = 统计权重（要用标定后的 σ，误差尺度才正确）。
        # 标定 σ 直接乘进信任域会收窄笼子，对远距/实际工况档反而致命。
        if sigma_prior is not None:
            sig_p = torch.as_tensor(np.asarray(sigma_prior, np.float64),
                                    dtype=torch.float32, device=dev)
        elif sigma_norm is not None:
            sig_p = sig
        else:
            sig_p = None

        sc = torch.log(torch.tensor(max(imax * 0.3, 1.0), device=dev)
                       ).requires_grad_(True)           # 总标度
        Bj = torch.tensor(0.01, device=dev).requires_grad_(True)   # Debye-Waller B
        jit = torch.zeros(self.n_ring, device=dev).requires_grad_(True)
        logs = torch.full((self.n_ring,), float(np.log(0.007)), device=dev
                          ).requires_grad_(True)
        logbg0 = torch.log(torch.tensor(max(imed * 0.5, 0.5), device=dev)
                           ).requires_grad_(True)
        logbg1 = torch.log(torch.tensor(max(imed * 0.01, 0.05), device=dev)
                           ).requires_grad_(True)
        logah = torch.log(torch.tensor(max(imed * 0.1, 0.5), device=dev)
                          ).requires_grad_(True)
        qh = torch.tensor(1.5, device=dev).requires_grad_(True)
        logsh = torch.log(torch.tensor(0.25, device=dev)).requires_grad_(True)
        nui = [sc, Bj, jit, logs, logbg0, logbg1, logah, qh, logsh]

        def clamp_nui():
            with torch.no_grad():
                Bj.clamp_(self.dw_lo, self.dw_hi)
                jit.clamp_(self.jit_lo, self.jit_hi)
                logs.clamp_(float(np.log(sigma_range[0])), float(np.log(sigma_range[1])))
                logbg0.clamp_(None, float(np.log(900.0)))
                logbg1.clamp_(None, float(np.log(600.0)))
                qh.clamp_(self.hq[0], self.hq[1])
                logsh.clamp_(float(np.log(self.hs[0])), float(np.log(self.hs[1])))

        # ---- 阶段A：几何冻结，暖机 nuisance ----
        opt_nui = torch.optim.Adam([{"params": nui, "lr": lr_nui}])
        with torch.no_grad():
            q0 = self.q_map(torch.stack(pns).detach())
        for _ in range(warmup):
            mu = self._render(q0, sc, Bj, jit, logs, logbg0, logbg1, logah, qh, logsh)
            loss = (mu[mask] - y[mask] * torch.log(mu[mask])).mean()
            opt_nui.zero_grad()
            loss.backward()
            opt_nui.step()
            clamp_nui()

        # NN 几何处拟合分（守门基准）
        with torch.no_grad():
            mu = self._render(q0, sc, Bj, jit, logs, logbg0, logbg1, logah, qh, logsh)
            loss_init = float((mu[mask] - y[mask] * torch.log(mu[mask])).mean())

        # ---- 阶段B：联合优化（几何各参数独立学习率，余弦衰减）----
        groups = [{"params": [pns[i]], "lr": lr_geo * float(lr_mult[i]),
                   "lr0": lr_geo * float(lr_mult[i])} for i in range(6)]
        groups.append({"params": nui, "lr": lr_nui * 0.5, "lr0": lr_nui * 0.5})
        opt = torch.optim.Adam(groups)
        mask_b = mask
        y_m = y[mask]
        loss_trace = []
        best = dict(loss=float("inf"), pn=None, nui=None)
        stale = 0
        for it in range(iters):
            cf = 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * it / iters))
            for g in opt.param_groups:
                g["lr"] = g["lr0"] * cf
            q = self.q_map(torch.stack(pns))
            mu = self._render(q, sc, Bj, jit, logs, logbg0, logbg1, logah, qh, logsh)
            if it == 100:                # 离群点剔除（热点/坏点）
                with torch.no_grad():
                    out = y > mu + 10.0 * torch.sqrt(mu) + 20.0
                    mask_b = mask & (~out)
                    y_m = y[mask_b]
            loss = (mu[mask_b] - y_m * torch.log(mu[mask_b])).mean()
            if soft_prior > 0.0 and sig_p is not None:
                pen = (((torch.stack(pns) - pn0) / (sig_p + 1e-6)) ** 2).mean()
                loss = loss + soft_prior * pen

            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for i in range(6):                       # σ-信任域夹紧
                    pns[i].clamp_(pn_lo[i], pn_hi[i])
            clamp_nui()

            lv = loss.item()
            loss_trace.append(lv)
            if lv < best["loss"]:
                best = dict(loss=lv, pn=[p.detach().clone() for p in pns],
                            nui=[t.detach().clone() for t in nui])
                stale = 0
            else:
                stale += 1
            if it >= 60 and stale >= 25:
                break
            if verbose and (it + 1) % 50 == 0:
                print(f"  it{it+1:4d}  loss={lv:.4e}")

        out_pn = (torch.stack(best["pn"]) if best["pn"] is not None
                  else torch.stack([p.detach() for p in pns]))
        # 纯似然（不含 soft_prior 罚项）：双轨比较与跨配方比较的公平基准
        if best["nui"] is not None:
            with torch.no_grad():
                mu_b = self._render(self.q_map(out_pn), *best["nui"])
                pure_loss = float((mu_b[mask_b] - y_m * torch.log(mu_b[mask_b])).mean())
        else:
            pure_loss = best["loss"]
        return dict(
            params=self.pn_to_phys(out_pn),
            params_init=np.asarray(params_phys_init, np.float64),
            loss=best["loss"], loss_init=loss_init, loss_trace=loss_trace,
            pure_loss=pure_loss,
            iters=len(loss_trace), n_pix=n_pix, sec=time.time() - t0,
        )

    # ------------------------------------------------------------ 双轨模式
    def refine_dual(self, img_u16: np.ndarray, params_phys_init: np.ndarray,
                    sigma_norm=None, trust_k: float = 1.0, soft_prior: float = 0.1,
                    iters: int = 400, adopt_nats: float = 1.0,
                    sigma_prior=None, trust_floor=None, **kw) -> dict:
        """双轨精修：笼内轨 + 同起点拆笼轨，纯似然好 > adopt_nats 才采纳 B。

        动机（probes/refine_diag.py 2026-10-26）：
          - far AgBh / 部分近距样本的粗解误差 > σ-信任域半径，真值在笼外
            （笼子病）→ 拆笼轨能走回真值（334mm→0.44mm 实测）
          - 同起点拆笼在 10 样本上救 3 个零误伤；跨大距离多起点才有
            "更深假谷"风险（#0856），双轨不跨距离起点，风险可控
          - 比较用 pure_loss（不含罚项），阈值 adopt_nats=1 nat 防假谷误采纳

        sigma_norm 传【已标定】σ（×T）效果最佳（笼紧 + 尾部由 B 轨兜底）。
        """
        A = self.refine(img_u16, params_phys_init, iters=iters,
                        sigma_norm=sigma_norm, trust_k=trust_k,
                        soft_prior=soft_prior, sigma_prior=sigma_prior,
                        trust_floor=trust_floor, **kw)
        B = self.refine(img_u16, params_phys_init, iters=iters,
                        sigma_norm=None, soft_prior=0.0, **kw)
        adopt = bool(B["pure_loss"] < A["pure_loss"] - adopt_nats)
        out = B if adopt else A
        out["dual_adopt"] = adopt
        out["pure_A"] = A["pure_loss"]
        out["pure_B"] = B["pure_loss"]
        out["sec_dual"] = A["sec"] + B["sec"]
        return out
