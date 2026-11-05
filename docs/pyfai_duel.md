# pyFAI 对照（duel）——同图同 λ 同标样，标准流程对决

**2026-11-05 · 探针 `probes/_pyfai_duel.py`（+`_pyfai_duel3.py` ceria 增量）·
数据 `probes/results/pyfai_duel_v3.md/.json`（v1/v2 原档保留）**

"pyFAI 对照"= 同一个问题（同图、同检测器含 spline、同波长、同标样身份），
让 pyFAI 用它自己的标准自动化流程解一遍，比谁准。

## 协议（三臂，信息对等）

| 臂 | 内容 | 初值 |
|---|---|---|
| **T** | 我们：`solver_core.solve_image` 统一入口（默认档 520 + R2 零样本 fallback）+ T7 原生精化（zoo 标准行口径；ceria 行=锁表 8 环生产口径 + rg2） | 无（零样本） |
| **P0** | pyFAI：`RingExtraction`（pyFAI-calib2 的自动找环/指派内核）+ `GeometryRefinement.refine3`，两轮重提取（"Recalibrate"迭代的无 GUI 等价） | 零起点：中心=几何中心，D∈{0.1,0.3,1.0}m 网格取其 chi2 最优（不看 GT） |
| **P1** | 同 P0，但以 T 解为初值 | T 的 (D, 中心) |

- λ：GT poni 有波长的行，T 与 P 臂都锁 λ（pyFAI `fix=["wavelength"]`）；
  **Frelon2k/halfccd 的 GT poni 无波长** → λ 联合模式（T=联合 λ–D；P 臂
  λ 初值 1Å 放开精修）。
- 指标：ΔD(mm)；**ΔλD(%)=λ̂·D̂ 相对 GT 的偏差**——λ–D 简并脊上 ≈0 表示
  几何正确、λ̂ 只是吸收了 GT 标称波长的偏差；Δc 双口径（GT / 共心真值
  T9——ceria 仪器 GT 中心系统偏 15–18px，T9 已证）。
- 标样映射：AgBh→`AgBh`、CeO2→`CeO2`、Al2O3→`alpha_Al2O3`、LaB6→`LaB6`。

## 结果（v3 总表见 probes/results/pyfai_duel_v3.md，此处要点）

**P0（pyFAI 标准自动流，零起点）：11/11 全部失败**——ΔλD 从 −87% 到
+369%，或精修原地震荡。机制：RingExtraction 的环**指派以初值几何预测
环位**，初值错 → 环标号错 → 小角近似下 D–环号简并 → SLSQP 收敛到
"自信但错"的极小值（chi2 还很小，AgBh 谐波标样 d_n=d₁/n 下简并精确
成立，D 被锁死在初值）。这就是 pyFAI-calib2 做成人机向导（人工点环/
校核初值）的原因；无 GUI 的"标准流程"并不存在零起点能力。

**P1（同起点，纯精修机器对比）**：
- λ-已知 zoo 行：P1≈T（在我们初值上重提取自洽，D 不再改善也不劣化，
  P1M 中心 2.3→1.6px 微赢）。
- ceria 行：D 打平（T −0.06/−0.01/+0.11 vs P1 −0.11/+0.20/−0.32mm）；
  **中心 pyFAI 略胜**（ceria400：14.6 vs 31.3pxᴳᵀ / 15.8 vs 26.9pxᶜᵒⁿ）——
  其逐点加权的 SLSQP 在中心维度比我们画布级残差锋利；我们的 fit_tilt
  下游在 D 上持平（−0.34/−0.08/+1.79mm），中心受 GT 污染口径影响
  （6–27pxᴳᵀ）。
- **pyFAI 主场胜出行（如实记）**：halfccd **P0=+0.53mm/3.8px** 完胜
  T（−3.87mm/55.5px）——spline 检测器（Frelon 畸变）+ 正确标样下，
  pyFAI 的前向模型是完整的；我们不吃 spline 是已声明的域界
  （docs/zoo_benchmark_v0：Frelon/halfccd 两行）。Frelon2k P0 −5.74mm/
  184.7px 同理（D 好、中心仍偏）。
- **坏初值对 pyFAI 同样致命**：Frelon2k P1（从我们无 spline 的 T 解
  出发）崩到 Δc=13287px——它的机器对初值的敏感不亚于我们，只是失败
  模式不同（我们：域界；它：环指派雪崩）。

**T（发货管线）**：zoo λ-已知行 ΔD −0.10…+0.01mm（与 zoo v0 一致：
P2M −0.05、E4M −0.03、E9M +0.01）；mar +5.87mm 是 λ̂ 脊（ΔλD −0.85%
= 几何正确，λ̂ 0.3611 吸收标称差）；P1M +55.8mm 全部由 GT 标称 λ 偏
3.2% 解释（zoo gt_selfcheck 记录 s=1.032，ΔλD +3.52% 正比）。ceria
生产行 ΔD ≤0.11mm（λD 口径 ≤0.04%）。

**耗时**：P0 2–6s/网格、P1 2–172s、T 1–542s（mar 的 λ̂ fallback 网格
是大头）。pyFAI 找点只在预测环带内做，天然比我们全画布盲扫快——但快
的前提是已有好初值，零起点下快而无用。

## pyFAI 2025.12 官方 bug（已绕道）

`RingExtraction._calculate_num_of_points_to_keep` 把 `units.CHI_RAD`
按**位置参数**塞进 `center_array(shape, unit, ...)` 的 shape 位 →
`numpy.indices` 崩（`'Unit' object is not iterable`）——**该版本自动
标定 API 全灭**。`__init__` 里同一函数用关键字传参是对的。补丁
（按签名传 shape + 修 clip 越界 off-by-one）见 `_pyfai_duel.py` 头部，
上游修复前必须保留。可报 issue。

## 过程教训（v1→v3）

1. **v1 的 cell() 把原始 D 印成 ΔD**（ceria150 显示"+148.82"实为
   D=148.82 vs GT 149.04=−0.22mm 正解）——被与 zoo 已知答案行（P2M
   −0.05 等）交叉核对抓出。长表探针的显示层也要冒烟已知答案。
2. **Frelon2k/halfccd GT poni 无 wavelength**——`ai.wavelength×1e10`
   直接 None×float 崩；zoo 对这两行走无 λ 联合分支，duel 必须同构。
3. ceria 锁表模式不带 2D 尾 → T7 门不触发 → v2 的 T 臂中心停在画布
   级；v3 补 rg2 后 ΔλD ≤0.04%。代表"发货管线"的探针必须带上发货
   下游。
4. pyFAI 的弃用告警（wavelength 函数化）会把 stderr 混进控制台日志，
   重定向时用 `2>$null`/分离落盘。

## 复现

```powershell
cd D:\pyfai-project
python -X utf8 -u probes\_pyfai_duel.py > probes\results\pyfai_duel_v2_console.txt 2>&1
python -X utf8 -u probes\_pyfai_duel3.py   # ceria 行 T+rg2 / P1 重初始化 → v3
```

## 一句话

零样本盲解（无初值、无人点环）这件事 pyFAI 标准流做不了——它的能力
边界是"好初值 + 人机向导"；同起点精修 D 维度与我们打平、中心维度
略胜、spline 域是真主场；我们赢在"无人值守、从零到亚毫米"。
