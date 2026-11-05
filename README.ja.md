<div align="center">

# pyfai-zero-shot

**粉末リング画像一枚から行う 2 次元 X 線検出器の<br>位相ブラインド・ゼロショット幾何較正**

🚫 キャリブラント物相の事前知識不要 &nbsp;·&nbsp; 🚫 幾何初期値不要 &nbsp;·&nbsp; 🚫 波長も未知で可<br>
✅ 標準 pyFAI PONI を出力 &nbsp;·&nbsp; ✅ ブートストラップ不確かさ付き

[English](README.md) · [简体中文](README.zh-CN.md) · [正體中文](README.zh-TW.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [Français](README.fr.md) · **日本語**

[![License: MIT](https://img.shields.io/badge/ライセンス-MIT-1a365d)](LICENSE)
[![Version](https://img.shields.io/badge/バージョン-v0.5_preprint-1a365d)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-306998?logo=python&logoColor=white)]()
[![Built for pyFAI](https://img.shields.io/badge/対象-pyFAI-013243)]()
[![Status](https://img.shields.io/badge/状態-論文執筆中-e67e22)]()

</div>

---

## なぜ必要か

標準的な較正ツール（pyFAI `calib2`、DAWN、FIT2D）は、**既知のキャリブラント、
手動でのリング選択、十分に良い幾何初期値**を前提としています。コールド
スタートで実行すると、自己整合的だが*誤った*解——リング指数/距離の縮退——
にロックされ、小さな残差が誤差を覆い隠します。下記の公開ベンチマークでは、
ゼロスタートの pyFAI は **11/11** フレームすべてに失敗します。本パイプライン
は、試料の物相を知らされないまま、単一フレームから無人でその大半を解きます。

## 仕組み

粉末相ライブラリ上で動作する**クラシックファースト・カスケード**を、
**シード間コンセンサスゲート**が仲裁します：

1. 高速なピーク位置ソルバーがまずカンバス中心から起算します
   （15 ms/フレーム）；
2. 失敗した場合のみフォールバックシードが作動——閉形式の放射対称中心
   推定器、そのオフ軸バイアスを補償する光線外挿シード、およびオプション
   として回帰ニューラルネットワーク；
3. 各シードが投票し、独立したシードによって再発見された解が勝利します——
   疑似解は排除されます；
4. 傾き精密化 + ブートストラップ光子リサンプリング → 各量の不確かさを
   伴う標準 **PONI** を出力します。

<div align="center">
<img src="assets/fig1_architecture.png" alt="パイプライン構成" width="820">
</div>

## 主要結果

| ベンチマーク | 結果 |
|---|---|
| CeO₂ キャリブラント · 7 距離、150–450 mm | 平均・最悪 \|ΔD\| = **0.1 mm**、7/7 収束；クラシック経路 15 ms/フレーム |
| Al₂O₃ · 52 静的フレーム | 51/52 を解決（暗いフレーム 1 枚を正しく棄却）；ばらつき **MAD 0.45 mm**；ブートストラップ σ(D) = **0.47 mm**、真値カバー率 6/6 |
| 装置横断 · 10 公開画像（8 枚が評価可能） | 物相同定 **6/8**；\|ΔD\| ≤ 0.1 mm が **4/8**；傾き ≲ 5 mrad；*全ての失敗に原因帰属* |
| 独立検証 | コールドスタートの pyFAI は同一の **11/11** フレームで失敗；我々の解から始めれば一致——利得は初期化にあり、異なるエンジンではありません |

<div align="center">
<img src="assets/fig2_frames_rings.png" alt="ベンチマーク画像とフィッティングリング" width="760"><br>
<sub>ベンチマーク画像とフィッティングリング：(a) 300 mm での CeO₂（8 本の一致リング）；(b) 52 静的 Al₂O₃ フレームの平均。</sub>
</div>

<details>
<summary><b>さらなる証拠</b>（CeO₂ 距離誤差 · pyFAI との独立比較）</summary>

<div align="center">
<img src="assets/fig3_ceria_errors.png" width="520"><br>
<sub>ネイティブ 2D 精密化による CeO₂ 7 距離の誤差：平均・最悪 |ΔD| = 0.1 mm。</sub><br><br>
<img src="assets/fig8_duel_ablation.png" width="760"><br>
<sub>pyFAI との独立比較（ゼロスタート P0 vs 本パイプライン T、同一スタート P1）とコンポーネントアブレーション。</sub>
</div>

</details>

## クイックスタート

```bash
pip install numpy scipy matplotlib pyFAI fabio pillow h5py torch
```

```python
import fabio
from probes.solver_core import solve_image

img = fabio.open("your_frame.edf").data.astype(float)
img[img >= 1e9] = 0.0            # センチネル/負値のマスク（zoo 方式）
img[img < 0] = 0.0

res = solve_image(img, px_m=172e-6)      # lam=None → λ–D 同時求解
print(res["phase"], res["sol"])          # 物相同定 + 幾何解
```

論文で参照している公開テスト画像
（<https://www.silx.org/pub/pyFAI/testimages/>）を `data/real/` に
ダウンロードし、各プローブを実行してください——`probes/results/` の
全テーブルが決定論的に再現されます（シード固定）。

| エントリーポイント | 目的 |
|---|---|
| `probes/solver_core.py` | 統一入口 `solve_image()`（カンバス選択 + ゼロショット安全フォールバック） |
| `probes/zoo_run.py` | 装置横断ベンチマーク（silx テスト画像） |
| `probes/phase_map.py` | 条件マップ（露光 / 傾き / スパン） |
| `probes/ablation8.py` | コンポーネントアブレーション（ストレスセット） |
| `probes/_pyfai_duel.py` | pyFAI との独立比較 |
| `generators/train.py` | NN 初期化器の学習（合成フレーム） |

## リポジトリ構成

```
torchfai/      ソルバーライブラリ（相、シミュレータ、精密化、モデル）
probes/        ベンチマークプローブ、作図スクリプト、GUI ソルバー
  results/     完全な結果表（.md/.json/.txt）+ 参照 PONI
generators/    NN 学習データ生成器 + マージツール
docs/          技術的エビデンス文書（実験ログ）
manuscript/    論文草稿 v0.5（Markdown）
assets/        本 README で使用する図
```

## ブラインドテストの招待

> どのビームラインからでも **1–2 枚の過去フレーム + 参照 .poni** を
> 募集中です——検出器、キャリブラント、距離、波長、年代は問いません。
> 画像と既知の較正値を送っていただければ、完全にブラインドで解析し、
> ΔD、ビーム中心、傾き、そのまま使える PONI、不確かさ区間、物相同定を
> お返しします。参照値は、我々の解が固定されタイムスタンプが押される
> まで一切見ません。**否定的な結果もそのまま報告します**——失敗には
> 言い訳ではなく原因の帰属が伴います。

**連絡先：** Ziao Liu · ziaoliu902@gmail.com

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

## ライセンスと謝辞

MIT（[LICENSE](LICENSE) 参照）。ベンチマーク画像は公開配布されている
pyFAI テスト画像（silx.org、D2AM/BM02、ESRF で取得）です。本プロジェクト
は pyFAI の独立した利用者であり、pyFAI チームとは提携関係にありません。

<sub>草稿（v0.5）は執筆中のものであり、査読を受けていません。多言語 README
は手動で保守されており、英語版が正となります。</sub>
