# Stage B：FwuSow 0–21 → FwuSow 0–31

本資料夾保存 **Stage B：分階段遷移學習延伸實驗** 結果。

Stage B 的主要目的，是在 Stage A 已完成：

```text
Kaggle Chick Weight 0–21
→ FwuSow Poultry 0–21
```

之後，進一步將 Stage A 所得模型權重延伸至：

```text
FwuSow Poultry 0–31
```

以檢驗「先完成 0–21 日齡對齊，再延伸至 0–31 後期日齡」的分階段遷移策略是否能改善 FwuSow 0–31 預測表現。

---

## 1. 實驗目的

Stage B 主要回答以下問題：

> 在 Stage A 已先完成 Kaggle 0–21 → FwuSow 0–21 日齡對齊後，將該權重延伸到 FwuSow 0–31 是否能比直接從零訓練更穩定？

本階段設計重點為：

1. 檢查 Stage A 權重能否延伸至 FwuSow 0–31。
2. 評估分階段遷移是否能改善 22–31 日齡後期延伸任務。
3. 比較 Without TL、Freeze 與 Partial Fine-tuning。
4. 與 Stage C-1 direct transfer 比較，判斷分階段遷移是否較直接遷移穩定。

---

## 2. 實驗流程

```text
Stage A：
Kaggle Chick Weight 0–21
→ FwuSow Poultry 0–21

Stage B：
Stage A best model
→ FwuSow Poultry 0–31
```

完整流程如下：

```text
Kaggle 0–21 Source Pre-train
        ↓
FwuSow 0–21 Target Fine-tuning
        ↓
FwuSow 0–31 Target Extension
```

---

## 3. 資料設定

| 項目 | 內容 |
|---|---|
| Source / Pre-model | Stage A best model |
| Target dataset | `FwuSow_Poultry_0_31_reuse0_21_scaler` |
| Target age range | 0–31 days |
| Extension range | 22–31 days |
| Scaler strategy | reuse FwuSow 0–21 scaler |
| Prediction target | daily mean body weight |
| Original unit | gram, g |

---

## 4. Scaler 設定說明

本階段使用：

```text
FwuSow_Poultry_0_31_reuse0_21_scaler
```

其意義為：

> 使用 FwuSow 0–21 日齡資料所建立的 scaler，去 transform FwuSow 0–31 日齡資料。

此設計目的在於保持 Stage A 與 Stage B 之間的 normalized scale 一致，使 Stage A 在 FwuSow 0–21 學得的權重能夠延續至 FwuSow 0–31 任務。

需要注意的是，22–31 日齡屬於延伸區間，部分 normalized values 可能超出 0–21 scaler 的原始範圍。因此，本階段結果需解讀為後期日齡延伸任務，而不是單純的同分布測試。

---

## 5. 模型設定

| 項目 | 設定 |
|---|---|
| Model | LSTM |
| Input window | `period = 3` |
| Forecast horizon | `horizon = 1` |
| Validation ratio | `valid_ratio = 0.3` |
| Seed | `1234` |
| Evaluation mode | `test` |
| Main scale | original scale after inverse transform |
| Metrics | MAE, MSE, RMSE, R² |

模型架構與 Stage A 相同：

```text
Input
→ TimeDistributed(Dense)
→ LSTM 1
→ BatchNormalization 1
→ LSTM 2
→ BatchNormalization 2
→ Dense output
```

---

## 6. 實驗組別

| Folder / Mode | 正式名稱 | 說明 |
|---|---|---|
| `without-transfer-learning` | Without TL | FwuSow 0–31 從零訓練 |
| `transfer-learning (Freeze)` | TL（Freeze） | 載入 Stage A 權重，凍結主要遷移層 |
| `transfer-learning (Unfreeze)` | TL（Partial FT） | 載入 Stage A 權重，進行部分微調 |

> 注意：若資料夾名稱為 `Unfreeze`，論文中仍建議稱為 **TL（Partial Fine-tuning / Partial FT）**，避免被理解為 full fine-tuning。

---

## 7. 主要結果

| Mode | MSE (g²) | RMSE (g) | MAE (g) | R² |
|---|---:|---:|---:|---:|
| Without TL | 4,311,058.519 | 2,076.309 | 2,069.960 | -123.996 |
| TL（Freeze） | **205,345.313** | **453.150** | **391.468** | **-4.954** |
| TL（Partial FT） | 205,469.715 | 453.288 | 391.633 | -4.957 |

---

## 8. 結果摘要

Stage B 結果顯示：

1. Without TL 在 FwuSow 0–31 延伸任務中表現非常不穩定。
2. 使用 Stage A 權重後，TL（Freeze）與 TL（Partial FT）均大幅降低 MAE 與 RMSE。
3. Freeze 與 Partial FT 結果幾乎一致，其中 Freeze 略優。
4. 雖然誤差大幅降低，但 R² 仍為負值，因此不能宣稱模型已完整掌握 0–31 後期日齡變異。
5. 本階段較適合判定為 **partial positive transfer**。

---

## 9. 與 Stage C-1 的關係

Stage C-1 是直接遷移：

```text
Kaggle 0–21 → FwuSow 0–31
```

Stage B 是分階段遷移：

```text
Kaggle 0–21 → FwuSow 0–21 → FwuSow 0–31
```

在相同 `reuse0_21_scaler` 條件下，Stage B 的 RMSE 約為 453 g，低於 Stage C-1 的約 832 g。這表示：

> **分階段遷移比直接遷移更穩定。**

因此，Stage B 可作為支持「先做 Target early adaptation，再延伸至 full target task」的實驗證據。

---

## 10. 結論

Stage B 的正式結論建議為：

> Transfer Learning 明顯改善 FwuSow 0–31 的預測誤差，其中 TL（Freeze）略優於 TL（Partial FT）。然而，由於 R² 仍為負值，模型尚未完整掌握 22–31 日齡後期變異。因此，本階段應定位為 **部分正向遷移** 與 **後期日齡延伸任務限制分析**，而非完整正向遷移成功。

---

## 11. 建議搭配檔案

本資料夾建議包含以下檔案：

```text
README.md
results_summary_StageB.md
without-transfer-learning/
transfer-learning (Freeze)/
transfer-learning (Unfreeze)/
```

各子資料夾內建議保留：

```text
params.json
log.txt
metrics_original_scale.txt
original_metrics_params.json
prediction_compare.csv
epoch_log.csv
prediction_normalized_scale.png
prediction_original_scale.png
yy_plot_normalized_scale.png
yy_plot_original_scale.png
Residual Plot.png
Error Histogram.png
architecture.png
best_model.hdf5
```

---

## 12. 論文使用建議

此 Stage B 結果適合放在：

1. 第三章：分階段遷移學習實驗設計。
2. 第四章：FwuSow 0–31 延伸任務結果分析。
3. Discussion：後期日齡外推限制、樣本數限制與 scaler 策略限制。
4. Appendix：完整圖表、prediction_compare.csv 與 residual analysis。

不建議單獨作為「完整正向遷移成功」證據；Stage A 才是主要跨資料集 Positive Transfer Learning 證據。
