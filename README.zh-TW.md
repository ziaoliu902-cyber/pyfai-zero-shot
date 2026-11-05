<div align="center">

# pyfai-zero-shot

**由單張粉末環影像進行的二維 X 射線偵測器幾何零樣本校準<br>（相位盲 · 零樣本）**

🚫 無需已知標樣物相 &nbsp;·&nbsp; 🚫 無需幾何初始值 &nbsp;·&nbsp; 🚫 波長可為未知<br>
✅ 直接輸出標準 pyFAI PONI &nbsp;·&nbsp; ✅ 內建 bootstrap 不確定度

[English](README.md) · [简体中文](README.zh-CN.md) · **正體中文** · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Français](README.fr.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-1a365d)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.5_preprint-1a365d)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/built%20for-pyFAI-013243)]()
[![Status](https://img.shields.io/badge/status-論文撰寫中-e67e22)]()

</div>

---

## 為什麼做這件事

傳統校準工具（pyFAI `calib2`、DAWN、FIT2D）預設**已知標樣、手動挑環、幾何初始值
夠好**。冷啟動（零初始值）執行時，它們會鎖進自洽但**錯誤**的解——環指標/距離簡併
——殘差很小，誤差被完美隱藏。在下方公開基準上，零初始值 pyFAI **11/11 全部失敗**；
本管線在無人介入、單張影像、不告知樣品物相的條件下解出其中大多數。

## 工作原理

在粉末相庫上執行的**經典優先級聯**，由**跨種子共識閘**仲裁：

1. 快速峰位求解器先從畫布幾何中心起解（15 毫秒/幀）；
2. 僅在失敗時啟用後備種子——閉式徑向對稱中心估計器、補償其偏軸收縮偏差的
   射線外推種子、以及可選的迴歸神經網路；
3. 各種子投票，被獨立種子重新發現的解勝出，從而剔除偽解；
4. 傾角精化 + bootstrap 光子重採樣 → 輸出帶各量不確定度的標準 **PONI**。

<div align="center">
<img src="assets/fig1_architecture.png" alt="管線架構" width="820">
</div>

## 一眼看結果

| 基準 | 結果 |
|---|---|
| CeO₂ 標樣 · 7 個距離，150–450 mm | 平均與最差 \|ΔD\| = **0.1 mm**，7/7 收斂；經典路徑 15 毫秒/幀 |
| Al₂O₃ · 52 張靜態幀 | 51/52 解出（1 張暗幀被正確拒絕）；幀間散布 **MAD 0.45 mm**；bootstrap σ(D) = **0.47 mm**，真值覆蓋率 6/6 |
| 跨儀器 · 10 張公開影像（8 張可評估） | 物相識別 **6/8**；\|ΔD\| ≤ 0.1 mm 佔 **4/8**；傾角 ≲ 5 mrad；*每個失敗都有歸因* |
| 獨立對照 | 零初始值 pyFAI在同樣 11 幀上**全部失敗**；從我們的解出發它與我們一致——增益來自初始化，而非不同的精化引擎 |

<div align="center">
<img src="assets/fig2_frames_rings.png" alt="基準幀與擬合環" width="760"><br>
<sub>基準幀與擬合環：(a) CeO₂，300 mm（8 條匹配環）；(b) 52 張靜態 Al₂O₃ 幀的平均。</sub>
</div>

<details>
<summary><b>更多證據</b>（CeO₂ 距離誤差 · 與 pyFAI 的獨立對決）</summary>

<div align="center">
<img src="assets/fig3_ceria_errors.png" width="520"><br>
<sub>CeO₂ 七距離誤差（原生二維精化）：平均與最差 |ΔD| = 0.1 mm。</sub><br><br>
<img src="assets/fig8_duel_ablation.png" width="760"><br>
<sub>與 pyFAI 的獨立對照（零初始值 P0 vs 本管線 T，同起點 P1）及組件消融。</sub>
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
img[img >= 1e9] = 0.0            # 哨兵值/負值遮蔽（zoo 口徑）
img[img < 0] = 0.0

res = solve_image(img, px_m=172e-6)      # lam=None → λ–D 聯合求解
print(res["phase"], res["sol"])          # 物相識別 + 幾何解
```

下載論文所引用的公開測試影像（<https://www.silx.org/pub/pyFAI/testimages/>）
至 `data/real/`，然後執行各探針——`probes/results/` 中的每張表都可確定性重現
（隨機種子固定）。

| 入口 | 用途 |
|---|---|
| `probes/solver_core.py` | 統一入口 `solve_image()`（畫布選擇 + 零樣本安全回退） |
| `probes/zoo_run.py` | 跨儀器基準（silx 測試影像） |
| `probes/phase_map.py` | 條件圖（曝光 / 傾角 / 跨度） |
| `probes/ablation8.py` | 組件消融（壓力集） |
| `probes/_pyfai_duel.py` | 與 pyFAI 的獨立對照 |
| `generators/train.py` | NN 初始化器訓練（合成幀） |

## 儲存庫結構

```
torchfai/      求解器函式庫（物相、模擬器、精化、模型）
probes/        基準探針、圖表腳本、GUI 求解器
  results/     完整結果表（.md/.json/.txt）+ 參考 PONI
generators/    NN 訓練資料產生器 + 合併工具
docs/          技術證據文件（實驗日誌）
manuscript/    論文草稿 v0.5（Markdown）
assets/        本 README 所用圖表
```

## 盲測邀請

> 我們正在向各實驗站徵集 **1–2 張歷史幀 + 對應參考 .poni**——不限偵測器、標樣、
> 距離、波長、年代。您寄送影像與已知校準；我們完全盲解，回傳 ΔD、束流中心、
> 傾角、可直接使用的 PONI、不確定度區間與物相識別。在我們的解固定並打上時間戳
> 之前，我們不會看到參考值。**負結果同樣如實回報**——失敗附帶歸因，而非藉口。

**聯絡人：** Ziao Liu · ziaoliu902@gmail.com

## 引用

```bibtex
@software{pyfai_zero_shot_2026,
  author = {Liu, Ziao},
  title  = {pyfai-zero-shot: phase-blind, zero-shot geometry calibration
            of 2-D X-ray detectors from powder rings},
  year   = {2026},
  url    = {https://github.com/ziaoliu902-cyber/pyfai-zero-shot},
  note   = {v0.5}
}
```

## 授權與致謝

MIT（見 [LICENSE](LICENSE)）。基準幀為 pyFAI 公開發布的測試影像
（silx.org，擷取自 ESRF D2AM/BM02 實驗站）；本專案為 pyFAI 的獨立使用者，
與 pyFAI 團隊無隸屬關係。

<sub>手稿（v0.5）為撰寫中的草稿，尚未經同儕審查。多語言 README 採人工維護，
以英文版為準。</sub>
