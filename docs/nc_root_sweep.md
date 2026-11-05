# ext1 root 杂项清扫（9 对 img+poni，V3 零样本 vs 官方 poni 直读）

跳过 mock/moke.tif（合成）与 orientation（1对多歧义）。λ/px 先验取自官方 poni。

| 对 | λ(Å) | V3 相 | Δc(px) | ΔD(mm) | ref D(mm) | V3 D(mm) |
|---|---|---|---|---|---|---|
| Eiger4M | - | ERR | - | - | - | - |
| Pilatus1M | 1.0000 | AgBh | 0.8 | +55.8 | 1583.2 | 1639.0 |
| Fairchild | - | FAIL | - | - | 88.2 | - |
| halfccd | - | ERR | - | - | - | - |
| LaB6_0020 | - | ERR | - | - | - | - |
| mar3450 | 0.3738 | Quartz | 1032.6 | +239.7 | 222.5 | 462.2 |
| LaB6_18keV | 0.9999 | Quartz | 910.5 | -178.7 | 229.2 | 50.5 |
| Pilatus6M | 1.0332 | Al2O3 | 113.0 | -203.6 | 300.0 | 96.4 |
| Eiger9M | 0.3738 | CeO2 | 0.7 | +0.0 | 213.2 | 213.2 |

V3 ok 5/9

phase: {'FAIL': 1, 'CeO2': 1, 'Al2O3': 1, 'Quartz': 2, 'ERR': 3, 'AgBh': 1}


## 补跑（修复路径/spline，剔除 LaB6_18keV 无真值行）

| 对 | λ(Å) | V3 相 | Δc(px) | ΔD(mm) | ref D(mm) | V3 D(mm) |
|---|---|---|---|---|---|---|
| Eiger4M | 0.9218 | Al2O3 | 0.3 | -0.0 | 162.6 | 162.5 |
| halfccd | - | ERR | - | - | - | - |
| LaB6_0020 | - | ERR | - | - | - | - |

## 总判读（root 清扫终版）

- **完美 2 对**：Eiger4M（Δc 0.32px/ΔD -0.03mm，λ=0.9218Å）、Eiger9M（Δc 0.66px/ΔD +0.01mm，
  **λ=0.3738Å 短波反例**→修正 ceria 判读的 λ 域界说法）。
- **部分 1 对**：Pilatus1M @1.58m 长距离——中心完美 0.78px 但 ΔD +55.8mm（3.5%），
  长 D 小环域 D-环径简并部分退化；相鉴 AgBh 待考。
- **败 3 对**：Fairchild（无解）；mar3450（Δc 1033px，λ=0.3738Å 图像板+LaB6 相鉴错）；
  Pilatus6M（Δc 113px，但官方 poni dist=0.3 整数疑似占位标定，真值存疑不计数）。
- **不可评 3 对**：halfccd / LaB6_0020（两 poni 同文件且无 Wavelength 字段，λ 先验无从取得——
  V3 需 λ 输入，判不可评；spline px 46.84/48.42µm 各向异性已从 spline 头读出备用）；
  LaB6_18keV（LaB6_3.poni 配对为猜测，λ 矛盾，剔除）。
- E2 战绩表可进：Eiger4M + Eiger9M（2 帧完美）；Pilatus1M 计"中心通过/D 退化"半票；
  其余不进。
