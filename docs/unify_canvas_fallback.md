# 画布策略与 fallback 门控统一（18-1104 需求 R1-R5）

生成：2026-11-05 ｜ 探针：`probes/_r1_shapes.py`、`_r2_calib.py`、
`_r2_calib2.py`、`_r2_gate_test.py` ｜ 代码：`solver_core.canvas_of /
solve_image`

## 0. 问题

同一算法、同一图，三处各写一套画布与门控，给出不同答案：

| 入口 | 画布 | fallback 门 | 病灶 |
|---|---|---|---|
| phase_fit / gui_solver | `to_canvas` 520 | 无 | — |
| zoo_run 主路径 | `to_canvas` 520 | **"相错 AND ps<0.55"**（依赖 GT 相签） | 零样本无此信号 |
| zoo_run._fallback_joint | ⌈maxdim/2⌉ | 上述 GT 门触发 | — |
| ablation8 v2 | `zoo_run.canvas_of` 723 | **裸 ps<0.55** | 误伤：胜项 ps 实测 0.446-0.476 |

后果（`results/ablation8v2_*.md`）：P2M −12.39 / Eiger4M +244.7 mm
（真实数据消融表不可发布），GUI 与基准同图不同解。

## 1. R1 画布策略（`solver_core.canvas_of`）

- **默认档** `tier="default"`：`max_side=520`——生产回归口径
  （ceria 0.1/0.1 mm、alumina 116.79/MAD 0.39、zoo 四胜全部建立在此）。
- **升级档** `tier="large"`：`max_side=⌈maxdim/2⌉`——仅 solve_image 在
  R2 门触发时使用；旧 `zoo_run.canvas_of(723)` 作废（723 档无回归基线，
  是 ablation8 v2 病灶的一部分）。
- 三处（zoo_run / phase_fit / gui_solver）+ ablation8 全部改调
  `canvas_of`/`solve_image`，禁止再各写一套（grep 验证：全库无第二处
  自定义 max_side 调用，除本文件与探针）。

各数据集两档池化倍率（`_r1_shapes.py`）：

| 数据 | 尺寸 | s_默认 | s_升级 | 扫描前提 |
|---|---|---|---|---|
| ceria(P2M 类) | 2527×2463 | 5 | 2 | 锁表模式→c 门不参与 |
| alumina | 1024² | 2 | 2 | 同 s → 永不扫 |
| Pilatus1M | 1043×981 | 3 | 2 | s<5 → 不扫 |
| Pilatus2M | 1679×1475 | 4 | 2 | s<5 → 不扫 |
| Eiger4M(_v2) | 2167×2070 | 5 | 2 | c 门候选 |
| Eiger9M | 3269×3111 | 7 | 2 | c 门候选 |
| Frelon2k/halfccd | 2048²/1024×2048 | 4 | 2 | s<5 → 不扫 |
| Pilatus6M(域界) | 2527×2463 | 5 | 2 | 自检失败不求解 |
| Fairchild(域界) | 4096² | 8 | 2 | 面积>3.5Mpx² 不扫 |
| mar3450 | 3450² | 7 | 2 | c 门候选（环合并假解） |

## 2. R2 门控（`solve_image` 内部，零样本安全）

### 2.1 触发（扫升级档的条件）

- **(a)** 默认档 None；
- **(b)** 默认档 `rms>FB_RMS_GATE(3.5)` 且 `m<FB_M_GATE(3)`；
- **(c)** 默认档 `s≥FB_S_RISK(5)` 且非锁表模式（q_list）。

前置否决：两档同 s（alumina）或升级档面积 > `FB_AREA_MAX(3.5Mpx²)`
（Fairchild）→ 永不扫。

### 2.2 (c) 两段式与分叉判据（对需求原案的修正）

需求原案 (c) 采用条件"ps 提升 ≥0.05 且 m 不降"。校准
（`_r2_calib2.txt`，2026-11-05）**两头推翻**：

| 数据 | 默认档 | 升级档（同 λ 单点，全质量） | D 分叉 |
|---|---|---|---|
| Eiger4M | Al2O3 m=7 rms=0.20 **ps=0.446** | Al2O3 m=7 rms=0.68 **ps=0.600** | **0.2%** |
| Eiger9M | CeO2 m=7 rms=0.54 **ps=0.476** | CeO2 m=9 rms=1.37 **ps=0.721** | **0.07%** |
| mar3450 | **Quartz 假解** m=11 rms=1.91 ps=0.327 | λ₀ 单点=AgBh 假解 ps=0.310；λ 网格=LaB6 m=5 ps≈0.39 | **≈108%**（假解 462 vs 真解 221mm） |

- E4M/E9M 大档是**同解重解**：D 与默认档一致（环是同一套环，只是画布
  分辨率不同），ps 虚涨 +0.15/+0.25 来自逐环相关在更细画布上的分辨率
  换算，不是画布档冲突。原案会把两条胜行劫持到无回归基线的大档解。
- mar3450 救援解 m=5 < Quartz 假解 m=11，原案"m 不降"会把唯一需要救的
  行误拒。
- **D-分叉是唯一同时分开两类的零样本判据**：两类相隔两个数量级以上
  （0.07-0.2% vs 108%）→ 取 `FB_FORK=0.10`。

实现：**(c) 先廉价分叉预检**——大画布单 λ 解（cascade/refine_2d 关；
**auto_center 必须开**：E4M 束在画布一角，几何中心经典解出垃圾 D →
假分叉 → 误劫持，`_r2_gate_test` 第一版实证；带定心后探针 E4M 16.8s /
E9M 58.3s / mar ~90s，远低于全质量大档解 41/194/230s）——
`|D_lrg/D_def − 1| ≥ 0.10` 才升级到全质量 λ 网格救援；探针失败或环<3
视为无分叉（大画布更没救，别浪费 7 点网格）。

### 2.3 救援扫描与采用

- (a)/(b)/(c-分叉确认) 统一走：λ 给定 → `geomspace(0.95,1.05,7)×λ`
  网格（T4 实证步长 1.68%，mar 盆地宽 1.4% 必有格点落盆；zoo 辅助模式
  传 `fb_grid=(0.96,1.04,9)` 锁相网格）；λ 未知 → 大画布联合
  solve_auto_lambda。
- 采用：T4 键序 `(ps, m, −rms)` 严格优于默认档（(a) 默认档 None 时
  有效即采用）。键序保 mar 通行（ps 首位 0.39>0.327 分胜负，m=5<11 不
  误杀）；E4M/E9M 根本进不了扫描（分叉预检拒）。
- `FB_PS_MARGIN=0.05` 留档不参与判决（需求原案常数，校准推翻）。

### 2.4 各数据集实测判决（`_r2_gate_test2.txt`，2026-11-05）

| 数据 | 门 | 结果 | 耗时 |
|---|---|---|---|
| Pilatus1M | 无（s=3） | AgBh m=3 ps=0.858，默认档 | 0.8s |
| Pilatus2M | 无（s=4） | CeO2 m=8 D=799.9mm，默认档 | 2.2s |
| Eiger4M | c 预检拒收（同解重解） | 默认档 Al2O3 D=162.5mm ✓ | 16.8s |
| Eiger9M | c 预检拒收 | 默认档 CeO2 D=213.2mm ✓ | 58.3s |
| mar3450 | c 预检分叉 ≥10% → 网格救援 | **fallback LaB6 m=5 rms=1.40 ps=0.587 λ̂=0.3611** ✓ | 626.8s |
| Frelon2k(联合) | 无（s=4；m=3 不满足 b 的 m<3） | CeO2 m=3 λ̂=1.1286（v3 行原样） | 68.1s |
| halfccd(联合) | 无（s=4） | Quartz m=9 λ̂=0.3054（v3 行原样） | 82.3s |
| ceria（锁表 q8） | c 不参与；b 不触发（rms<1.8） | 默认档（`phase_fit.py ceria` 回归验证） | — |
| alumina | 同 s 不扫 | 默认档（`phase_fit.py alumina` 回归验证） | — |

## 3. R3 单一入口（`solver_core.solve_image`）

```
solve_image(img, px_m_native, lam=None, want="auto", phases=None,
            q_list=None, i_rel=None, dbg=False, solve_kw=None,
            fb_grid=None) → dict(phase, sol, lam, lam_in, canvas, s,
            px_m, used_fallback, fb_reason, ps, rms, m, def_*, lrg_*)
```

- 盲解（lam 或 None）、锁相（phases=["X"]）、锁表（q_list=ceria 预注册
  8 强环 + rms_floor=1.8）三模式；solve_kw 透传
  cascade/auto_center/refine_2d/rms_floor。
- 调用方：`zoo_run.main`（有 λ / 无 λ 两分支 + 辅助模式 fb_grid）、
  `phase_fit`（ceria 锁表 / alumina 锁相 / lab6 锁相 / 抽样盲解）、
  `gui_solver.SolveWorker`（盲解 + 画布随 dict 下传 plot_radial）、
  `ablation8.solve_arm`（A-G 全臂）。

## 4. R4 回归锁（2026-11-05 实测全绿）

- **zoo（`results/_zoo_v4_run.txt`）10/10 行逐格一致**：四胜行
  P2M −0.05 / Eiger4M −0.03 / v2 −0.10 / Eiger9M +0.01；mar3450
  fallback 行 LaB6 m=5 rms=1.4、λ̂=0.3611、**ΔD@λ̂ +5.87**、
  s=2·λ̂·2Dⁿ、Δc(tilt) 2.4px；Frelon2k（CeO2✗ m=3，λ̂1.1286，−48.68）
  / halfccd（Quartz✗ m=9，λ̂0.3054，−3.87）域界行原样；P6M/Fairchild
  域界原样；P1M +55.80 原样。唯一可见变化：mar λ̂ 格新增 σ231% 简并
  标注（refine_lambda_d NameError 修复后 LSQ 收口真的跑了、被 σ 门
  拒收，λ̂ 值不变）。
- **`phase_fit.py ceria`**：平均 0.1 / 最差 0.1 mm ✓通过（7/7，
  r2d+2D 全链路不变）。
- **`phase_fit.py alumina`**：静态 MAD 0.39、平均帧 D=116.79mm
  （Δ−0.07）✓；逐帧 51/52（dark_0001 非数据帧恒失败，历史同）。
- **`ablation8.py` v3（`results/ablation8v3_20261105_0110.md`）**：
  A 臂与 zoo 逐格一致——**Pilatus2M −0.05、Eiger4M −0.03**（v2 病灶
  −12.39/+244.7 消失）；9 数据集 × 7 臂 **零 fallback 采用**（门全
  静默）；E 臂（喂 GT 相）同 A；F 臂联合 λ–D E4M −0.02。
  顺带修 v2 加载器：裸 float32 不清哨兵像素 → Eiger4M.edf 脏像素毁掉
  默认档解 → 分叉预检假阳性 → 救援采用 +244.7 垃圾（T7 哨兵教训的
  镜像：不清哨兵与清错哨兵一样掉错盆地）。现与 zoo_run.load_img 同
  口径（float64 + ≥1e9 清零 + 负值清零）。
- **`_t_spl_reg.py` 验收 B** ✓：向导口径 2 帧 ×9 字段逐项与 2026-11-03
  基线一致——同时验证 gui_solver 改走 solve_image 后向导行为不变
  （ceria_400 D=399.5687、ceria_150 判词 0.045° 合格）。

### R1 完整性审计（to_canvas 残留分类）

生产四入口（zoo_run / phase_fit / gui_solver / ablation8）全部经
solve_image；残留 to_canvas 调用分三类，均非门控面：
① 历史取证探针（`_t*`/`_mytake*`/`_r2d*` 等，冻结证据）；
② 分析工具走默认档同策略（bootstrap_uncertainty / sensitivity_probe /
  ablation8_syn 合成表 / poni_out 演示函数——synthetic 管道永不触发
  救援，无 fallback 语义）；
③ gui_solver.plot_radial 的旧 res dict 兜底。

## 5. R5 预算（实测修正）

- 默认档单帧 ≤0.4s（P1M 0.8s / P2M 2.2s 含多相评分与 2D 精化，与 v3
  持平）；**无门触发时零额外成本**（P1M/P2M/Frelon2k/halfccd/ceria/
  alumina 六类不扫）。
- c 预检（仅 s≥5 大图）：E4M 16.8s / E9M 58.3s——比"原案直接扫两档"
  的全质量大档解（41/194s）省 2-3 倍，且换来正确拒收。
- 分叉确认后的 λ 网格救援：mar 627s（探针 ~90 + 7 点全解）。**对
  18-1104 需求原案"fallback ≤8s"的偏离**：3450² 画布上单次全质量
  solve_auto 就 ~75s，7 点网格物理下界即 ~500s——原案预算对 mar 级
  大图不成立，两段式设计已是该算法下的最小成本（预检把 7 点网格压到
  仅分叉确认后执行）。触发即罕见（zoo 10 行中 1 行）。

## 6. 顺带修复

- zoo_run 用 `refine_lambda_d` 未 import（NameError 被 try 吞）——mar λ̂
  一直取网格点 0.966×λ₀ 而非 LSQ 收口点；已补 import，σ 门照旧拒收
  简并角域，行值不受影响（回归验证）。
