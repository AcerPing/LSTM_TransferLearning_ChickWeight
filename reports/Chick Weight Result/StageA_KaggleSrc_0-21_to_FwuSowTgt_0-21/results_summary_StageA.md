# results_summary_StageA.md

# Stage A：Kaggle Source 0–21 → FwuSow Target 0–21

## 0. 本檔案定位

本檔案彙整 **LSTM Transfer Learning** 於 **StageA_KaggleSrc_0-21_to_FwuSowTgt_0-21** 之實驗結果，作為本研究中「日齡對齊式跨資料集遷移學習」的主要整理文件。

Stage A 的核心目的，是在 Source Data 與 Target Data 皆限制於 **0–21 日齡** 的條件下，檢驗以 **Kaggle Chick Weight Dataset** 預訓練所得之 LSTM 權重，是否能改善 **FwuSow Poultry Dataset** 的平均體重預測效果。

本階段可作為後續 Stage B / Stage C 及 xLSTM 實驗的 LSTM baseline 與 feasibility study。Stage A 是目前福壽資料中最乾淨、最容易防守的跨資料集 Positive Transfer Learning 證據。

---

## 1. 實驗目的

Stage A 主要回答以下研究問題：

> 在 Source Data 與 Target Data 皆限制於 0–21 日齡範圍的條件下，Kaggle Chick Weight Source Data 的 LSTM 預訓練權重，是否能有效遷移至 FwuSow Poultry Target Data，並相較於 Without Transfer Learning baseline 改善目標資料集的雞隻平均體重預測表現？

本階段設計重點如下：

1. 將 Source Data 與 Target Data 限制於共同日齡區間 0–21 days。
2. 降低 Source / Target 日齡範圍不一致造成的干擾。
3. 檢驗 LSTM 權重是否能從公開資料集遷移至實際場域資料集。
4. 作為後續 0–31 日齡延伸任務與 xLSTM 實驗之比較基準。

---

## 2. 資料與模型設定

| 項目 | 設定 |
|---|---|
| Source Data | Kaggle Chick Weight Dataset |
| Source age range | 0–21 days |
| Target Data | FwuSow Poultry Dataset |
| Target age range | 0–21 days |
| Input features | min, max, range, p25, p75, IQR, median, std |
| Prediction target | mean |
| Output unit after inverse transform | gram (g) |
| Model | LSTM baseline |
| Period / Time Window | 3 |
| Horizon | 1 |
| Main seed | 1234 |
| Validation ratio | 0.3 |
| Main reported scale | Original scale after inverse transform |

---

## 3. 實驗組別

| No. | Experiment | 說明 |
|---:|---|---|
| 1 | Pre-train | 使用 Kaggle Source Data 進行 source-domain pre-training，作為後續遷移學習權重來源。 |
| 2 | Without Transfer Learning | FwuSow Target Data 從零訓練，作為 Target baseline。 |
| 3 | Transfer Learning (Freeze) | 載入 Kaggle 預訓練權重，凍結遷移層，主要訓練 output layer。 |
| 4 | Transfer Learning (Partial Fine-tuning / Partial FT) | 載入 Kaggle 預訓練權重，允許部分後段層微調。此組為 Stage A 最佳結果。 |

> 註：原始資料夾或程式輸出中可能使用 `Unfreeze` 命名；但若實際 trainable parameters 並非全層可訓練，論文與簡報建議統一稱為 **TL（Partial Fine-tuning / Partial FT）**，避免誤導為 Full Fine-tuning。

---

## 4. Positive Transfer Learning 判斷標準

本研究中，Positive Transfer Learning 的判斷基準如下：

> 若 Transfer Learning 模型在 FwuSow Target Data 上，相較於 Without Transfer Learning baseline 具有較低的 MAE、MSE、RMSE，且 R² 明顯提升或由負轉正，則可判定遷移學習對目標任務具有正向幫助。

Stage A 屬於較強的正向遷移判定，因為：

1. Transfer Learning 組別的 MAE / MSE / RMSE 均大幅低於 Without TL。
2. R² 由 Without TL 的負值轉為正值。
3. Freeze 與 Partial FT 兩種遷移策略皆優於 Without TL，其中 Partial FT 最佳。

---

## 5. Original-scale Metrics 結果總表

評估尺度：inverse-transformed original scale  
MAE / RMSE 單位：g  
MSE 單位：g²  
R² 無單位

| Mode | MSE (g²) | RMSE (g) | MAE (g) | R² | 判讀 |
|---|---:|---:|---:|---:|---|
| Pre-train（Kaggle Source） | 3.080 | 1.755 | 1.597 | 0.976 | Source 預訓練成功；僅作為權重來源，不與 Target 組直接比較。 |
| Without TL（FwuSow Target） | 171,083.974 | 413.623 | 395.418 | -15.801 | Target 從零訓練不穩定，作為必要 baseline。 |
| TL（Freeze） | 5,088.740 | 71.335 | 59.149 | 0.500 | 明確優於 Without TL，顯示來源權重具有幫助。 |
| **TL（Partial FT）** | **3,050.515** | **55.231** | **50.158** | **0.700** | **Stage A 最佳結果；形成明確 Positive Transfer Learning 證據。** |

---

## 6. 指標改善幅度

以 Without TL 作為 FwuSow Target baseline，比較最佳 Transfer Learning 組別 TL（Partial FT）：

| 指標 | Without TL | TL（Partial FT） | 改善幅度 |
|---|---:|---:|---:|
| MAE (g) | 395.418 | 50.158 | 約下降 87.31% |
| RMSE (g) | 413.623 | 55.231 | 約下降 86.65% |
| R² | -15.801 | 0.700 | 由負值轉為正值 |

此結果表示，在 0–21 日齡對齊條件下，Kaggle Source Data 所學得之早期生長趨勢表徵，經 Partial Fine-tuning 後可有效改善 FwuSow Target Data 的預測表現。

---

## 7. 各組結果判讀

### 7.1 Pre-train：Kaggle Source 0–21

Pre-train 組的作用是確認 LSTM 是否能在 Kaggle Source Data 上學得穩定的早期生長趨勢。其 original-scale 指標為 MAE = 1.597 g、RMSE = 1.755 g、R² = 0.976，顯示 Source model 具備作為後續 Transfer Learning 權重來源的基本條件。

但此結果僅代表模型於 Source domain 的學習能力，不能直接視為 FwuSow Target Data 的預測結果，也不能單獨用來判定 Positive Transfer Learning。Positive Transfer Learning 必須透過 Target task 上的 Without TL 與 TL 組別比較判定。

### 7.2 Without Transfer Learning：FwuSow 0–21 baseline

Without TL 組為 FwuSow Target Data 從零訓練的 baseline。此組 MAE = 395.418 g、RMSE = 413.623 g、R² = -15.801，顯示在目前小樣本、chronological split 與 period=3 設定下，LSTM 從零訓練難以穩定掌握 FwuSow 0–21 後段日齡的體重趨勢。

此結果不是失敗到不能使用，而是作為 Transfer Learning 改善幅度的必要對照組。若沒有此 baseline，就無法判斷 Kaggle → FwuSow 的遷移是否真正有效。

### 7.3 Transfer Learning（Freeze）

Freeze 組載入 Kaggle Source 預訓練權重，並凍結主要遷移層，只訓練少量輸出層參數。此組 MAE = 59.149 g、RMSE = 71.335 g、R² = 0.500，明顯優於 Without TL，顯示 Source pre-trained weights 即使在凍結條件下，仍能提供對 Target Data 有益的早期生長趨勢表徵。

Freeze 組可判定為明確正向遷移，但不是 Stage A 最佳結果。

### 7.4 Transfer Learning（Partial Fine-tuning / Partial FT）

Partial FT 組為 Stage A 最佳結果。其 MAE = 50.158 g、RMSE = 55.231 g、R² = 0.700，明顯優於 Without TL 與 Freeze 組。此結果表示，僅保留 Source model 的部分通用時序表徵並允許後段層微調，有助於模型適應 FwuSow Target Data 的資料特性。

因此，本組可作為 Stage A 的主要 Positive Transfer Learning 證據。

---

## 8. 圖表與輸出檔案建議

Stage A 各實驗資料夾建議保留以下輸出檔案，以利後續論文與簡報追溯：

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

正式簡報與論文中，建議優先使用：

1. `metrics_original_scale.txt` 或 `original_metrics_params.json`：作為主要數值來源。
2. `prediction_original_scale.png`：展示原始尺度下的預測趨勢。
3. `yy_plot_original_scale.png`：展示 observed vs predicted 的接近程度。
4. `residual_plot_original_scale.png`：輔助說明殘差分布與是否存在系統性偏差。
5. `epoch_log.csv` / Learning Curve：作為模型訓練過程與收斂狀態佐證。

---

## 9. 論文可用之保守結論

Stage A 結果顯示，在 Source Data 與 Target Data 皆限制於 0–21 日齡的條件下，Kaggle Chick Weight Source Data 預訓練所得之 LSTM 權重，能有效改善 FwuSow Poultry Target Data 的平均體重預測表現。相較於 Without Transfer Learning baseline，Transfer Learning 組別在 MAE、MSE、RMSE 與 R² 上皆有明顯改善；其中 TL（Partial Fine-tuning）為最佳結果，MAE 由 395.418 g 降至 50.158 g，RMSE 由 413.623 g 降至 55.231 g，且 R² 由 -15.801 提升至 0.700。此結果支持在日齡對齊條件下，Kaggle Source Data 所學得之早期生長趨勢表徵可遷移至 FwuSow Target Data，構成明確的 Positive Transfer Learning 證據。

---

## 10. 論文撰寫注意事項

1. **Pre-train 不應與 Target 組直接比較。**  
   Pre-train 是 Source domain validation result，功能是確認 Kaggle model 具備作為權重來源的條件。

2. **Positive Transfer Learning 判定應以 Target task 為準。**  
   真正的比較是 FwuSow Without TL vs FwuSow Transfer Learning。

3. **Unfreeze 建議改稱 Partial FT。**  
   若實際不是全層可訓練，正式論文與簡報應使用 TL（Partial Fine-tuning / Partial FT）。

4. **Stage A 可以稱為明確正向遷移，但不要過度延伸到 0–31。**  
   Stage A 證明的是 0–21 日齡對齊條件下的正向遷移，不等於已完整解決 FwuSow 0–31 後期日齡外推問題。

5. **測試樣本數有限，仍需保守表述。**  
   FwuSow 0–21 的測試資料點有限，結果應定位為 LSTM baseline / feasibility study，不宜宣稱已達實務部署水準。

---

## 11. 簡報口徑建議

Group Meeting 或口試可採用以下說法：

> Stage A 是本研究中最乾淨的跨資料集遷移實驗，因為 Kaggle Source Data 與 FwuSow Target Data 皆限制於 0–21 日齡。結果顯示，Transfer Learning 相較 Without TL 明顯降低 MAE 與 RMSE，且 R² 由負轉正。其中 Partial Fine-tuning 表現最佳，代表 Kaggle Source model 所學到的早期生長趨勢，經目標資料微調後能有效改善 FwuSow 0–21 的預測表現。因此，Stage A 可作為本研究中 Kaggle 0–21 → FwuSow 0–21 的 Positive Transfer Learning 主要證據。

---

## 12. 總結

Stage A 的核心結論為：

```text
Kaggle 0–21 → FwuSow 0–21
在日齡對齊條件下，LSTM Transfer Learning 明顯優於 Without Transfer Learning。
Partial Fine-tuning 為最佳策略，形成明確 Positive Transfer Learning 證據。
```

本階段結果應作為：

1. 福壽資料跨資料集 Positive Transfer Learning 的主要 LSTM baseline 證據。
2. 後續 Stage B / Stage C 延伸任務的起點。
3. 後續 xLSTM 實驗比較的基準。
4. 碩士論文中「智慧養殖實際場域案例」的重要支撐資料。
