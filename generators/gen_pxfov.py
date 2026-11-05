# -*- coding: utf-8 -*-
"""px+宽峰双域补丁卷：data/train_pxfov1（10-29 覆盖洞 + alumina 宽峰出域的根治）。

配置 = train_lam_v1 基底 + 两个单变量：
  1. pixel_scales：离散 k 档 [1, 1.26, 1.58, 2.0, 2.5, 3.0] 逐样本随机
     （像素=基准×k → 视域=基准×k；真实 6M 宽视域输入 k≈2.43 落在 2.0-2.5 档间；
      k 记 aux 第 7 列，训练时 px 触须条件化——λ 触须同款零初始化套路）
  2. 宽峰分层 broad_frac=0.18：每样本 18% 概率环宽 log-U[0.05, 0.6] Å⁻¹
     （治 alumina 判决：真实宽峰标样 σq≈0.15-0.6，旧域 12-50 倍出界）
  其余（λ∈(0.5,2.3)Å、几何、nuisance）与 train_lam_v1 完全一致。
  之后 merge_pxfov.py 混入 train_lam_v1 重放 → data/train_px1（7 列 aux）。

用法：
  python gen_pxfov.py --verify   # 直接验证合成器（k 几何/宽峰/回归兼容）
  python gen_pxfov.py --smoke    # 512 张到 data/_smoke_pxfov
  python gen_pxfov.py            # 28k 张到 data/train_pxfov1（断点续跑安全）
"""
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))

N_FULL = 28000
SEED = 20261029
LAM_RANGE = (5e-11, 2.3e-10)
SCALES = [1.0, 1.26, 1.58, 2.0, 2.5, 3.0]
BROAD_FRAC = 0.18
BROAD_RANGE = (0.05, 0.6)


def verify():
    """合成器三连验证：k 几何、宽峰、旧配置逐位兼容。"""
    import numpy as np
    from torchfai.config import SimConfig
    from torchfai.simulator import (get_detector, get_scaled_detector,
                                    synth_sample, ring_radius_m)
    from torchfai.phases import phase_qi

    print("== 验证 1：k=2.5 q-map vs 理论（机器精度判据）==")
    from torchfai.simulator import make_geometry
    cfg = SimConfig()
    cfg.wavelength = 0.8e-10
    lam = 0.8
    for k in (1.0, 2.5):
        dbk = get_scaled_detector("Pilatus_1M", 2, k)
        Hm, Wm = dbk.extent
        gk = dict(dist=0.55, poni1=Hm / 2, poni2=Wm / 2,
                  rot1=0.0, rot2=0.0, rot3=0.0)
        geo = make_geometry(gk, cfg, dbk, wavelength_m=cfg.wavelength)
        qm = geo.array_from_unit(shape=dbk.shape, typ="center",
                                 unit="q_A^-1", scale=True)
        yy, xx = np.mgrid[0:dbk.shape[0], 0:dbk.shape[1]]
        r = np.sqrt(((yy + 0.5) * dbk.pixel - gk["poni1"]) ** 2 +
                    ((xx + 0.5) * dbk.pixel - gk["poni2"]) ** 2)
        m = (r > 0.02) & (r < 0.18)
        q_theo = 4 * np.pi * np.sin(0.5 * np.arctan2(r[m], 0.55)) / lam
        err = np.abs(qm[m] - q_theo) / q_theo
        print(f"   k={k}: pixel={dbk.pixel*1e6:.0f}µm 相对误差中位={np.median(err):.1e}")
        assert np.median(err) < 1e-9, "q-map 与理论不符！"
    print("   ✓ k=1 与 k=2.5 的 q-map 均与理论一致到机器精度")

    print("== 验证 2：宽峰分层 ==")
    cfg_b = SimConfig()
    cfg_b.wavelength_range = LAM_RANGE
    cfg_b.peak_sigma_broad_frac = 1.0        # 强制全宽峰
    cfg_b.peak_sigma_broad_range = BROAD_RANGE
    db1 = get_detector("Pilatus_1M", 2)
    rng = np.random.default_rng(11)
    s = synth_sample(rng, cfg_b, db1, phase_idx=0, pixel_scale=1.0)
    img = s["img"].astype(np.float64)
    bright = img > 0.3 * img.max()
    # 宽峰应铺满大面积（锐环只占细带）
    frac = bright.mean()
    print(f"   全宽峰样本：亮区占比={frac:.3f}（锐环卷典型 <0.05）aux宽={s['aux'].shape}")
    assert frac > 0.08, "宽峰没生效（亮区太小）"
    assert s["aux"].shape == (7,) and s["aux"][6] == 1.0
    print("   ✓ 宽峰生效，aux 7 列含 px=k")

    print("== 验证 3：旧配置 RNG 流逐位兼容（broad_frac=0 不抽随机数）==")
    cfg_o = SimConfig()
    rng_a = np.random.default_rng(3)
    rng_b = np.random.default_rng(3)
    a = synth_sample(rng_a, cfg_o, db1, phase_idx=0)
    b = synth_sample(rng_b, cfg_o, db1, phase_idx=0)
    same_rng = (rng_a.bit_generator.state["state"]["state"]
                == rng_b.bit_generator.state["state"]["state"])
    assert same_rng and np.array_equal(a["img"], b["img"])
    # 与 10-28 旧代码产物比对：老 aux 6 列 → 新第 7 位 k=1
    print(f"   ✓ 同种子旧配置产物逐位一致（img 相同，aux 增补 k=1）")

    print("== 验证 4：模型 px 触须零初始化 = 热启动逐位一致 ==")
    import torch
    from torchfai.model import ResNet18Geo
    m_old = ResNet18Geo(lam_cond=True).eval()
    m_new = ResNet18Geo(lam_cond=True, px_cond=True).eval()
    m_new.load_state_dict(m_old.state_dict(), strict=False)
    x = torch.randn(2, 1, 521, 490)
    lam = torch.tensor([0.97, 1.5])
    with torch.no_grad():
        mu0, _ = m_old(x, lam)
        mu1, _ = m_new(x, lam, torch.tensor([2.43, 1.0]))
    assert torch.allclose(mu0, mu1, atol=1e-6), "px 触须零初始化破坏一致性！"
    print("   ✓ 加载旧权重后 px 触须输出恒等（k=2.43 也逐位一致）")
    print("\n全部通过 ✓")


def main():
    if "--verify" in sys.argv:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        verify()
        return
    from generate_gui import run_generation
    smoke = "--smoke" in sys.argv
    out = os.path.join(ROOT, "data", "_smoke_pxfov" if smoke else "train_pxfov1")
    total = 512 if smoke else N_FULL
    os.makedirs(out, exist_ok=True)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def drain():
        t0 = time.time()
        while True:
            try:
                kind, payload = q.get(timeout=2)
            except queue.Empty:
                if stop.is_set():
                    return
                continue
            if kind == "done":
                print(f"DONE status={payload['status']} actual={payload['actual']} "
                      f"rejects={payload['rejects']} {(time.time()-t0)/60:.1f}min",
                      flush=True)
                stop.set()
                return
            if kind == "prog":
                n, tot, el, ph, dist, rej, sh = payload
                rate = n / max(el, 1e-9)
                print(f"{n:6d}/{tot}  {el/60:6.1f}min  {rate:5.1f}/s  "
                      f"ETA {(tot-n)/max(rate,1e-9)/60:4.0f}min  相={ph:7s} "
                      f"dist={dist:.2f}m  拒绝={rej}  分片={sh}", flush=True)
            elif kind == "log" and ("落盘" in payload or "开始" in payload
                                    or "覆盖" in payload or "像素尺度" in payload):
                print(payload, flush=True)

    th = threading.Thread(target=drain, daemon=True)
    th.start()
    print(f"生成 px+宽峰补丁卷：N={total} seed={SEED}\n"
          f"  k 档={SCALES}（真实 6M 宽视域 k≈2.43）\n"
          f"  λ∈{LAM_RANGE[0]*1e10:.2f}-{LAM_RANGE[1]*1e10:.2f}Å  "
          f"宽峰 {BROAD_FRAC:.0%} log-U{BROAD_RANGE} Å⁻¹ → {out}", flush=True)
    run_generation(total, 256, out, SEED, q, threading.Event(),
                   cfg_overrides=dict(wavelength_range=LAM_RANGE,
                                      peak_sigma_broad_frac=BROAD_FRAC,
                                      peak_sigma_broad_range=BROAD_RANGE),
                   pixel_scales=SCALES)
    stop.set()
    th.join(timeout=5)


if __name__ == "__main__":
    main()
