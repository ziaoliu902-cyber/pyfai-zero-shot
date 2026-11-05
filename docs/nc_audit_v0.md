# NC 核心证据 v0：公开标定文件的共心真值审计

> ⚠️ **口径警示（2026-11-03 深夜纠偏，见 zoo_benchmark_v3.md T9 纠偏注）**：
> 本表 GTΔc 列以共心 ref 为基准，但锐度对决证明共心拟合在**带倾角/环被边切**的帧上
> 有 ~6px 量级偏置（σ 是精度不是准确度）。ceria/E9M/P2M 的 GT（Fit2D 口径）基本无辜；
> halfccd 的 56px（配置错配，T5 字节证据）与 P6M 的 113px（非标定帧）为定性结论。
> v0.1 协议：锐度仲裁 + tilt 分层 + λ̂ 双臂（mar3450/halfccd 待终裁）。

生成：_nc_audit.py ｜ 70 对 (image, poni) ｜ 协议：零样本求解（GT 仅评估侧）→ concentric_center 共心真值 → 双口径 Δc。λ=poni 名义值；px=poni 探测器规格。

| 标定文件 | 相(我们) | center_ref | 环数 | σ_ref(px) | 我们Δc(px) | **GTΔc(px)** | ΔD(mm) |
|---|---|---|---|---|---|---|---|
| ceria_150 |  CeO2✔ | (1227.9,1257.7) | 8 | 0.029 | **0.40** | **6.3** | -0.07 |
| ceria_200 |  CeO2✔ | (1227.4,1257.2) | 6 | 0.026 | **0.41** | **4.6** | -0.11 |
| ceria_250 |  CeO2✔ | (1225.3,1257.6) | 5 | 0.028 | **0.07** | **5.0** | -0.03 |
| ceria_300 |  CeO2✔ | (1224.9,1256.9) | 3 | 0.031 | **0.09** | **3.3** | -0.04 |
| ceria_350 |  CeO2✔ | (1223.9,1256.5) | 2 | 0.017 | **0.61** | **2.5** | +0.10 |
| ceria_400 |  CeO2✔ | (1226.8,1260.2) | 2 | 0.041 | **26.96** | **4.5** | +0.11 |
| ceria_450 |  CeO2✔ | (1220.4,1257.4) | 2 | 0.032 | **7.57** | **2.6** | -0.07 |
| Pilatus1M |  AgBh✔ | — | 2.3ᴳᵀ | +55.80 | ref不可用(环<2) |
| Pilatus2M |  CeO2✔ | (735.9,837.8) | 7 | 0.016 | **7.78** | **0.8** | -0.05 |
| Pilatus6M |  Al2O3✔ | (1227.8,1260.4) | 13 | 0.020 | **0.04** | **112.9** | -203.60 |
| Eiger4M |  Al2O3✔ | — | 5.7ᴳᵀ | -0.03 | ref不可用(环<2) |
| Eiger4M_v2 |  Al2O3✔ | — | 5.6ᴳᵀ | -0.10 | ref不可用(环<2) |
| Eiger9M |  CeO2✔ | (1572.2,1534.4) | 9 | 0.022 | **0.43** | **5.8** | +0.01 |
| Frelon2k | ERROR OSError: Spline File does not exist D:\pyfai-project\frelon.spline |
| halfccd | ERROR OSError: Spline File does not exist D:\pyfai-project\halfccd.spline |
| LaB6_0020poni | ERROR OSError: Spline File does not exist D:\pyfai-project\halfccd.spline |
| mar3450 |  Quartz✔ | (1763.1,676.7) | 2 | 0.189 | **0.57** | **1033.1** | +239.67 |
| Fairchild | SOLVE FAIL | — | — | — | — | — |
| alumina/al2o3_0000. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0001. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0002. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0003. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0004. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0005. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0006. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0007. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0008. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0009. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0010. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0011. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0012. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0013. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0014. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0015. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0016. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0017. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0018. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0019. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0020. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0021. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0022. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0023. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0024. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0025. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0026. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0027. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0028. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0029. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0030. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0031. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0032. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0033. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0034. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0035. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0036. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0037. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0038. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0039. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0040. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0041. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0042. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0043. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0044. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0045. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0046. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0047. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0048. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0049. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/al2o3_0050. | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |
| alumina/dark_0001.e | ERROR OSError: Spline File does not exist D:\pyfai-project\distorsion_2x2.spline |

**GT 审计**：N=11 文件，median **4.6px**、p90 112.9px、max **1033.1px**、>5px 占比 **45%**。
**我们**：N=11 帧，median **0.43px**、p90 7.78px、max 26.96px。
总耗时 2.5 分钟。