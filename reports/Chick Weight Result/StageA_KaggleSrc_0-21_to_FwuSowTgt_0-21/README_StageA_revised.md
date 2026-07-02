# Stage A：Kaggle Source 0–21 → FwuSow Target 0–21

本資料夾保存 **Stage A：日齡對齊式遷移學習** 實驗結果。

本階段實驗目的在於驗證：當 SourceData 與 TargetData 皆限制於 **0–21 日齡** 範圍時，使用 **Kaggle Chick Weight SourceData** 預訓練所得之 LSTM 權重，是否能改善 **FwuSow Poultry TargetData** 的雞隻平均體重預測效果。

此階段屬於本研究中最重要的 **age-aligned cross-dataset transfer learning baseline**，可作為後續 Stage B / Stage C 以及 xLSTM 比較實驗的基準。

---

## 1. 實驗目的

Stage A 主要回答以下問題：

> 在 SourceData 與 TargetData 日齡範圍皆為 0–21 days 的條件下，Kaggle Chick Weight 預訓練權重是否能遷移至 FwuSow Poultry TargetData，並相較於 Without Transfer Learning baseline 改善預測表現？

本階段的設計重點為：

1. 先將 SourceData 與 TargetData 限制於共同日齡區間 0–21 days。
2. 降低 Source / Target 日齡範圍不一致造成的干擾。
3. 檢驗 LSTM 權重是否能從公開資料集遷移至實際場域資料集。
4. 作為後續 0–31 日齡延伸任務與 xLSTM 實驗之比較基準。

---

## 2. 資料設定

| 項目 | 設定 |
|---|---|
| SourceData | Kaggle Chick Weight Dataset |
| Source age range | 0–21 days |
| TargetData | FwuSow Poultry Dataset |
| Target age range | 0–21 days |
| Input features | min, max, range, p25, p75, IQR, median, std |
| Prediction target | mean |
| Unit after inverse transform | gram |
| Period / Time Window | 3 |
| Horizon | 1 |
| Main seed | 1234 |
| Main reported scale | Original scale after inverse transform |

---

## 3. 實驗組別

本資料夾包含以下四組實驗結果：

| No. | Experiment | 說明 |
|---:|---|---|
| 1 | Pre-train | 使用 Kaggle SourceData 進行 source-domain pre-training |
| 2 | Without Transfer Learning | FwuSow TargetData 從零訓練，作為 Target baseline |
| 3 | Transfer Learning (Freeze) | 載入 Kaggle 預訓練權重，凍結遷移層，只訓練 output layer |
| 4 | Transfer Learning (Unfreeze / Partial Fine-tuning) | 載入 Kaggle 預訓練權重，允許部分後段層微調 |

---

## 4. 主要比較方式

Stage A 主要比較：

1. **Without Transfer Learning vs Transfer Learning (Freeze)**
2. **Without Transfer Learning vs Transfer Learning (Unfreeze / Partial Fine-tuning)**
3. **Freeze vs Unfreeze / Partial Fine-tuning**

其中：

- **Without Transfer Learning** 作為 FwuSow TargetData baseline。
- 若 Transfer Learning 組別相較 Without Transfer Learning 具有較低 MAE / MSE / RMSE，且 R² 明顯提升或轉正，則可作為 **Positive Transfer Learning** 的證據。

---

## 5. 結果摘要

Stage A 結果顯示，Transfer Learning 相較 Without Transfer Learning 具有明顯改善效果。

其中，**Transfer Learning (Unfreeze / Partial Fine-tuning)** 為本階段最佳結果：

| Metric | Original-scale result |
|---|---:|
| MAE_original | 50.157903 g |
| MSE_original | 3050.515185 g² |
| RMSE_original | 55.231469 g |
| R²_original | 0.700426 |

Normalized-scale evaluation 中，Partial Fine-tuning 結果亦為：

| Metric | Normalized-scale result |
|---|---:|
| MAE | 0.047381 |
| MSE | 0.002722 |
| RMSE | 0.052174 |
| R² | 0.700426 |

### Interpretation

- Without Transfer Learning 作為 FwuSow TargetData baseline。
- Transfer Learning (Freeze) 相較 Without Transfer Learning 可降低預測誤差，顯示 source pre-trained weights 對 TargetData 具有幫助。
- Transfer Learning (Unfreeze / Partial Fine-tuning) 在本階段整體表現最佳，R² 達 **0.700426**，已由原本 baseline 的負值轉為正值。
- 因此，Stage A 可作為 **Kaggle 0–21 → FwuSow 0–21 的 Positive Transfer Learning 證據**。

---

## 6. Output Files

各實驗資料夾可能包含以下檔案：

```text
best_model.hdf5
transferred_best_model.hdf5
epoch_log.csv
params.json
log.txt
Learning Curve.png
architecture.png
prediction_normalized_scale.png
yy_plot_normalized_scale.png
residual_plot_normalized_scale.png
error_histogram_normalized_scale.png
prediction_original_scale.png
yy_plot_original_scale.png
residual_plot_original_scale.png
error_histogram_original_scale.png
prediction_compare.csv
metrics_original_scale.txt
original_metrics_params.json
inverse_run_log.txt
inverse_output_file_list.txt
```

### Notes

- `log.txt` records normalized-scale metrics.
- `metrics_original_scale.txt` records inverse-transformed original-scale metrics used for thesis reporting.
- `prediction_compare.csv` contains both normalized-scale and original-scale prediction results.
- `*_original_scale.png` files are recommended for presentation and thesis reporting.
- `*_normalized_scale.png` files are mainly used for technical checking and supplementary analysis.

---

## 7. Research Positioning

Stage A 是本研究中最乾淨的跨資料集遷移實驗，因為 SourceData 與 TargetData 皆限制於 0–21 日齡，日齡範圍一致。

本階段結果可支撐以下論點：

> 在日齡對齊條件下，Kaggle Chick Weight SourceData 所學得之早期生長趨勢表徵，可遷移至 FwuSow Poultry TargetData，並有效改善目標資料集之平均體重預測表現。

後續 Stage B / Stage C 將以此階段結果為基礎，進一步檢驗模型於 FwuSow 0–31 延伸日齡任務中的穩定性與限制。
