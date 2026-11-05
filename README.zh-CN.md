<div align="center">

# pyfai-zero-shot

**从单张粉末环图像出发的二维 X 射线探测器几何零样本标定<br>（相位盲 · 零样本）**

🚫 无需已知标样物相 &nbsp;·&nbsp; 🚫 无需几何初值 &nbsp;·&nbsp; 🚫 波长可未知<br>
✅ 直接输出标准 pyFAI PONI &nbsp;·&nbsp; ✅ 自带 bootstrap 不确定度

[English](README.md) · **简体中文** · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Français](README.fr.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.5_preprint-1a365d)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/built%20for-pyFAI-013243)]()
[![Status](https://img.shields.io/badge/status-论文撰写中-e67e22)]()

</div>

---

## 为什么做这件事

传统标定工具（pyFAI `calib2`、DAWN、FIT2D）默认**已知标样、手动挑环、几何初值足够好**。
冷启动（零初值）运行时，它们会锁进自洽但**错误**的解——环指标/距离简并——残差很小，
误差被完美隐藏。在下方公开基准上，零初值 pyFAI **11/11 全部失败**；本流水线在无人
干预、单帧、不告知样品物相的条件下解出其中大多数。

## 工作原理

在粉末相库上运行的**经典优先级联**，由**跨种子共识门**仲裁：

1. 快速峰位求解器先从画布几何中心起解（15 毫秒/帧）；
2. 仅在失败时启用后备种子——闭式径向对称中心估计器、补偿其偏轴收缩偏差的
   射线外推种子、以及可选的回归神经网络；
3. 各种子投票，被独立种子重新发现的解胜出，从而剔除伪解；
4. 倾角精化 + bootstrap 光子重采样 → 输出带各量不确定度的标准 **PONI**。

<div align="center">
<img src="assets/fig1_architecture.png" alt="流水线架构" width="820">
</div>

## 一眼看结果

| 基准 | 结果 |
|---|---|
| CeO₂ 标样 · 7 个距离，150–450 mm | 平均与最差 \|ΔD\| = **0.1 mm**，7/7 收敛；经典路径 15 毫秒/帧 |
| Al₂O₃ · 52 张静态帧 | 51/52 解出（1 张暗帧被正确拒绝）；帧间散布 **MAD 0.45 mm**；bootstrap σ(D) = **0.47 mm**，真值覆盖率 6/6 |
| 跨仪器 · 10 张公开图像（8 张可评估） | 物相识别 **6/8**；\|ΔD\| ≤ 0.1 mm 占 **4/8**；倾角 ≲ 5 mrad；*每个失败都有归因* |
| 独立对照 | 零初值 pyFAI 在同样 11 帧上**全部失败**；从我们的解出发它与我们一致——增益来自初始化，而非不同的精化引擎 |

<div align="center">
<img src="assets/fig2_frames_rings.png" alt="基准帧与拟合环" width="760"><br>
<sub>基准帧与拟合环：(a) CeO₂，300 mm（8 条匹配环）；(b) 52 张静态 Al₂O₃ 帧的均值。</sub>
</div>

<details>
<summary><b>更多证据</b>（CeO₂ 距离误差 · 与 pyFAI 的独立对决）</summary>

<div align="center">
<img src="assets/fig3_ceria_errors.png" width="520"><br>
<sub>CeO₂ 七距离误差（原生二维精化）：平均与最差 |ΔD| = 0.1 mm。</sub><br><br>
<img src="assets/fig8_duel_ablation.png" width="760"><br>
<sub>与 pyFAI 的独立对照（零初值 P0 vs 本流水线 T，同起点 P1）及组件消融。</sub>
</div>

</details>

## 快速上手

```bash
pip install numpy scipy matplotlib pyFAI fabio pillow h5py torch
```

```python
import fabio
from probes.solver_core import solve_image

img = fabio.open("your_frame.edf").data.astype(float)
img[img >= 1e9] = 0.0            # 哨兵值/负值掩蔽（zoo 口径）
img[img < 0] = 0.0

res = solve_image(img, px_m=172e-6)      # lam=None → λ–D 联合求解
print(res["phase"], res["sol"])          # 物相识别 + 几何解
```

下载论文所引用的公开测试图（<https://www.silx.org/pub/pyFAI/testimages/>）
到 `data/real/`，然后运行各探针——`probes/results/` 中的每张表都可确定性复现
（随机种子固定）。

| 入口 | 用途 |
|---|---|
| `probes/solver_core.py` | 统一入口 `solve_image()`（画布选择 + 零样本安全回退） |
| `probes/zoo_run.py` | 跨仪器基准（silx 测试图） |
| `probes/phase_map.py` | 条件图（曝光 / 倾角 / 跨度） |
| `probes/ablation8.py` | 组件消融（压力集） |
| `probes/_pyfai_duel.py` | 与 pyFAI 的独立对照 |
| `generators/train.py` | NN 初始化器训练（合成帧） |

## 仓库结构

```
torchfai/      求解器库（物相、模拟器、精化、模型）
probes/        基准探针、图表脚本、GUI 求解器
  results/     完整结果表（.md/.json/.txt）+ 参考 PONI
generators/    NN 训练数据生成器 + 合并工具
docs/          技术证据文档（实验日志）
manuscript/    论文草稿 v0.5（Markdown）
assets/        本 README 所用图表
```

## 盲测邀请

> 我们正在向各线站征集 **1–2 张历史帧 + 对应参考 .poni**——不限探测器、标样、
> 距离、波长、年代。您发送图像与已知标定；我们完全盲解，返回 ΔD、束流中心、
> 倾角、可直接使用的 PONI、不确定度区间和物相识别。在我们的解固定并打上时间
> 戳之前，我们不会看到参考值。**负结果同样如实报告**——失败附带归因，而非借口。

## 引用

```bibtex
@software{pyfai_zero_shot_2026,
  author = {{The pyfai-zero-shot authors}},
  title  = {pyfai-zero-shot: phase-blind, zero-shot geometry calibration
            of 2-D X-ray detectors from powder rings},
  year   = {2026},
  url    = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot},
  note   = {v0.5; Zenodo DOI to be assigned}
}
```

## 许可与致谢

MIT（见 [LICENSE](LICENSE)）。基准帧为 pyFAI 公开发布的测试图
（silx.org，采集于 ESRF D2AM/BM02 线站）；本项目是 pyFAI 的独立使用者，
与 pyFAI 团队无隶属关系。

<sub>手稿（v0.5）为撰写中的草稿，尚未经过同行评审。多语言 README 手工维护，
以英文版为准。</sub>
