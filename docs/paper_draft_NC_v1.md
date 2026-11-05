# Paper Draft v1 — Nature Communications format（2026-10-27）

> v0 的中文工作稿保留在 paper_draft_v0.md；本稿按 NC Article 结构：
> 无编号引言（3 段内讲 broad）→ Results（4-6 个小标题）→ Discussion → Methods（文末详尽）。
> 语言最终为英文；本稿中文骨架 + 关键英文术语锚定。

---

## Title (draft)

**Uncertainty-aware neural coarse solution with differentiable physical refinement enables automatic geometry calibration of 2D diffraction detectors**

备选（更 broad）：*A dual-engine approach coupling calibrated deep uncertainty with differentiable physics for self-calibrating diffraction instrumentation*

## Abstract (≤200 words, no citations)

高通量同步辐射与自主实验平台中，二维面探的几何标定仍是专家手工瓶颈。We present a two-stage framework: an uncertainty-aware network learns a coarse solution and per-channel uncertainties from physics-based simulation; a differentiable Poisson-likelihood refiner then optimizes the six geometry parameters, with the predicted uncertainty shaping a trust region. We show that naive use of this uncertainty fails on a operationally realistic subset due to condition-dependent miscalibration of the rotation channels, and introduce a dual-track refinement that adopts the un-caged solution only when the pure likelihood improves by a decisive margin (>1 nat) — validated by paired sign tests on N=394 (distance p=2.9e-3, rotation p=4.2e-5). On real standard-sample data (CeO2, Pilatus 6M, seven calibrated distances), the physical layer converges from nominal geometry to median 1.06 mm (best -0.02 mm) under wavelength and detector substitution, while the network layer alone provably fails from a wavelength-distance degeneracy — motivating an explicit wavelength conditioning. 〔若 P1 完成：One wavelength-conditioned model generalizes across 0.5–2.3 Å without retraining.〕 The framework replaces minutes of expert ring-picking with seconds of automated computation.

## Introduction (no heading, ≤4 paragraphs)

1. **Broad context**: 面探衍射是材料/化学/地学高通量表征主干；所有下游积分依赖六参数几何；现行标定 = 专家拾环 + 迭代拟合（Fit2D/Dioptas/pyFAI），分钟-小时级/每次配置变化，自主实验/快速原位实验的瓶颈。〔ref Fit2D, pyFAI, Dioptas〕
2. **Gap**: 两类自动化尝试——纯回归网络（域脆、无物理一致性保证）与纯优化（初值敏感、多谷）——缺少把两者优点焊接起来的机制；尤其缺少**不确定度何时可信的判据**。
3. **Here we**: 提出双引擎框架；三个机制贡献：(i) 逐通道不确定度温度标定与**条件性失准**的发现；(ii) **纯似然判据的双轨采纳**——先验失准的物理自检；(iii) 仿真→真实跨波长/跨探测器验证 + λ-简并失效机理。
4. **Implications**: 范式可迁移至任何"可微物理渲染器 + NN 粗解"的仪器自校准问题。

## Results

### R1 An uncertainty-aware network provides coarse geometry in milliseconds
- ResNet18 + NLL head, 80k physics-simulated images (six standard phases, Poisson counting, full nuisance family); curriculum: far-distance remediation (AgBh) → near-distance all-phase remediation.
- Held-out N=4800: coarse distance MAE 7.48 mm; per-channel σ. 〔Fig.1 架构 + Fig.2 仿真样例 + 粗解精度分布〕

### R2 Calibrated uncertainty, and where it breaks
- Global per-channel temperature T on val: {dist .317, poni1 .532, poni2 .515, rot1 1.365, rot2 1.459, rot3 .997}.
- dist/poni uniformly conservative; **rot1/rot2 conditionally miscalibrated (~3× overconfident on operationally realistic subset)** — global post-hoc calibration cannot fix. 〔Fig.4 z-score 散点 ×子集〕
- Consequence: soft prior mis-specified → systematic convergence into shallow valleys (cage disease).

### R3 Dual-track refinement with a pure-likelihood adoption criterion
- Track A (caged) + Track B (same start, cage removed); adopt B iff pure likelihood improves >1 nat.
- Full-pool N=394 (operationally realistic): distance median 0.220→0.184 mm, <0.5 mm 67.8→69.5%, paired sign p=2.9e-3 (dist) / 4.2e-5 (rot2); adoption 21%, >1 mm regressions 1/394 vs recoveries 4. 〔Fig.3 双轨示意 + Fig.5 B 轨纯似然差分布〕
- Heavy tail (poni p90 ≈77 px) invariant to all σ post-processing → attributed to poni↔rot information degeneracy (single-image, single-phase limit). 〔补充简并可视化〕

### R4 Failure forensics: domain shift, replay, and a cautionary tale
- Normalization-constant misalignment masquerading as catastrophic forgetting (1311 mm artifact, exact reproduction) → deployment protocol.
- Finetune on operationally realistic subset: far-distance forgetting +48% (v4c); +6k far replay → -30% (26.7→18.6 mm) (v4d); net gain over base_v3 ≈ 0 → **domain shift is absorbed more efficiently by the refinement layer than by network retraining**.

### R5 Sim-to-real: millimetre convergence on real standard samples
- Benchmark: pyFAI MX_ceria (Pilatus 6M, λ=0.972 Å, CeO2, 7 distances, multi-frame ground truth ±0.07 mm); phase/λ/GT cross-validated (peak mismatch ≤0.002 Å⁻¹).
- Physics layer with nominal-geometry start (label distance + detector centre + zero tilt): median 1.06 mm, best -0.02 mm; ground-truth start is a fixed point; residual poni offset ±12.6 px = inter-model systematic.
- Network layer alone fails from **λ×dist degeneracy** (prediction loses monotonicity, −37%~+40%); image-space rescaling provably breaks phase fingerprints → wavelength must enter as conditioning, not preprocessing. 〔Fig.6 ceria 图+残差 + 简并示意〕

### R6 (P1 gated) One model, all wavelengths
- λ-conditioned training (λ∈[0.5,2.3] Å, same-geometry multi-λ binding): single model passes ceria (0.97 Å), AgBh@Pilatus1M (1.00 Å), alumina (0.71 Å) without retraining. 〔实验后填〕

## Discussion
- 双引擎分工：NN = 域内速度与起点；物理 = 跨域保底与收敛保证；不确定度 = 二者的通信协议，但必须配失准自检（纯似然判据）。
- Limitations: rot3 dormant; single-image degeneracy; simulated counting statistics (真实探测器响应/parallax 未建模); 标定真值间的模型间系统差。
- Outlook: wavelength conditioning; phase auto-identification as first-stage + safety gate; q-χ canonical representation; multi-distance joint calibration; autonomous beamline integration (3.7 s/frame throughput).

## Methods (NC 要求详尽，文末)
1. Physics-based generator（d-spacings, Poisson, nuisance, feasibility sampling; config table）
2. Network & training（arch, NLL, augmentation, curriculum schedules）
3. Differentiable refiner（Poisson NLL, trust region, soft prior, nuisance warm-up, outlier rejection, dual-track criterion; hyperparameter table）
4. Uncertainty calibration（temperature fitting, conditional miscalibration audit protocol）
5. Real-data benchmark & protocol（MX_ceria provenance, cross-validation of GT, nominal-start protocol, detector/λ injection）
6. Statistics（paired sign tests, N, effect sizes）
7. Data & code availability（仓库结构、一键复现命令表）

## Figures plan
- Fig.1 框架总览（双引擎 + 不确定度管道 + 双轨）
- Fig.2 仿真生成器样例网格（六相×距离）
- Fig.3 双轨机制与采纳判据（示意+纯似然差分布）
- Fig.4 σ 失准解剖（全局 T + 条件失准散点）
- Fig.5 仿真主结果（394 全量、配对检验、重尾）
- Fig.6 sim-to-real（ceria 实拍图、七档收敛、λ-简并失效与机理）
- Fig.7 (P1) λ-conditioned 跨波长泛化

## NC 差距清单（投稿前 must-do）
| 级 | 项 | 状态 |
|---|---|---|
| P0 | vs pyFAI calib2d head-to-head（精度/耗时/人工介入） | 未做 |
| P0 | real 库全量泛化（Pilatus1M-AgBh / Frelon-Al2O3 / LaB6-gonio） | 未做（库已备） |
| P1 | λ-conditioned 训练+三波长对拍 | 方案已定 |
| P1 | 吞吐量基准（自动化 vs 手工） | 数字已有（3.7s/张），需对照测量 |
| P2 | 物相自动识别 | 方案已定 |
| P2 | 英文全文 | 骨架就绪后 2 周 |

## 目标时间线（建议）
- 第 1 周：P0 两项 + 250 档全扫收尾
- 第 2 周：λ-conditioned 数据+训练（P1）
- 第 3 周：Fig.4-7 + 全表回填
- 第 4 周：英文全文 + 内审
