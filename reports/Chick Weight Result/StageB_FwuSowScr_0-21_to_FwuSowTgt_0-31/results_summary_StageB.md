# Stage B Results Summary：FwuSow 0–21 → FwuSow 0–31

## 0. 本檔案定位

本檔案整理 **StageB_FwuSowScr_0-21_to_FwuSowTgt_0-31** 之 LSTM Transfer Learning 實驗結果。

Stage B 的目的不是單獨證明模型已完整解決 FwuSow 0–31 日齡預測任務，而是檢驗：

> 在 Stage A 已完成 `Kaggle 0–21 → FwuSow 0–21` 日齡對齊遷移後，是否能將已適應 FwuSow 早期日齡的模型權重，進一步延伸至 FwuSow 0–31 日齡任務。

因此，本階段屬於 **分階段遷移學習（staged transfer learning）延伸實驗**，主要用來評估：

1. Stage A 權重是否有助於 FwuSow 0–31 後期日齡預測。
2. 分階段遷移是否優於直接從 Kaggle 0–21 遷移至 FwuSow 0–31。
3. Transfer Learning 是否能在誤差層面降低預測偏差。
4. 0–31 後期日齡任務是否仍存在外推與資料限制。

---

## 1. 實驗任務定義

| 項目 | 內容 |
|---|---|
| Stage | Stage B |
| 實驗名稱 | `StageB_FwuSowScr_0-21_to_FwuSowTgt_0-31` |
| Source / Pre-model | Stage A：`Kaggle 0–21 → FwuSow 0–21` |
| Target Data | `FwuSow_Poultry_0_31_reuse0_21_scaler` |
| Target 日齡 | 0–31 日齡 |
| 延伸區間 | 22–31 日齡 |
| 模型 | LSTM |
| 輸入特徵 | daily statistics：`min, max, range, p25, p75, IQR, median, std` |
| 預測目標 | `mean`，每日平均體重 |
| 評估尺度 | inverse-transformed original scale |
| 原始單位 | gram, g |

---

## 2. 實驗設計

Stage B 採用與 Stage A 相同的 LSTM 架構，並使用 `period = 3` 的 sliding window 設定，即以前 3 天的統計特徵預測下一日平均體重。

本階段的 Target Data 使用 `FwuSow_Poultry_0_31_reuse0_21_scaler`，也就是將 FwuSow 0–31 日齡資料放在 FwuSow 0–21 scaler 的尺度空間中。此設計目的在於維持與 Stage A 權重相同的 normalized scale，使 Stage A 已學得的 0–21 日齡表示能延伸至 0–31 日齡任務。

---

## 3. 實驗組別

| 組別 | 說明 | 研究用途 |
|---|---|---|
| Without TL | 直接在 FwuSow 0–31 上從零訓練 | Target baseline |
| TL（Freeze） | 載入 Stage A 權重，凍結主要遷移層，僅訓練輸出層 | 保守遷移策略 |
| TL（Partial FT） | 載入 Stage A 權重，允許部分後段層微調 | 部分微調策略 |

> 備註：若原資料夾命名為 `transfer-learning (Unfreeze)`，正式論文與簡報建議統一稱為 **TL（Partial Fine-tuning / Partial FT）**，避免被誤解為全層 fine-tuning。

---

## 4. 主要實驗參數

| 參數 | 設定 |
|---|---|
| `period` | 3 |
| `valid_ratio` | 0.3 |
| `seed` | 1234 |
| `eval_mode` | test |
| split rule | chronological split, no shuffle |
| output scale | original scale after inverse transform |
| main metrics | MAE, MSE, RMSE, R² |

---

## 5. Original-scale 指標總表

| Mode | MSE (g²) | RMSE (g) | MAE (g) | R² | 判讀 |
|---|---:|---:|---:|---:|---|
| Without TL | 4,311,058.519 | 2,076.309 | 2,069.960 | -123.996 | 從零訓練嚴重不穩定 |
| TL（Freeze） | **205,345.313** | **453.150** | **391.468** | **-4.954** | 誤差大幅下降，Stage B 最佳 |
| TL（Partial FT） | 205,469.715 | 453.288 | 391.633 | -4.957 | 與 Freeze 幾乎相同，略差 |

---

## 6. Stage B 結果判讀

### 6.1 Transfer Learning 明顯優於 Without TL

Without TL 在原始尺度下的 MAE 約為 2,069.960 g、RMSE 約為 2,076.309 g，R² 約為 -123.996。此結果顯示，若直接在 FwuSow 0–31 日齡任務中從零訓練 LSTM，模型難以穩定掌握後期日齡的生長趨勢。

相較之下，TL（Freeze）與 TL（Partial FT）皆將 MAE 降至約 391–392 g，RMSE 降至約 453 g。這代表 Stage A 權重對 Stage B 具有明顯誤差改善效果。

### 6.2 Freeze 與 Partial FT 幾乎沒有差異

Stage B 中，TL（Freeze）與 TL（Partial FT）的結果非常接近：

| 比較項目 | TL（Freeze） | TL（Partial FT） | 判斷 |
|---|---:|---:|---|
| MAE | 391.468 | 391.633 | Freeze 略優 |
| RMSE | 453.150 | 453.288 | Freeze 略優 |
| R² | -4.954 | -4.957 | Freeze 略優 |

此結果表示，在目前單一 seed、相同 scaler 與相同資料切分條件下，解凍更多參數並未帶來額外改善。對 FwuSow 0–31 後期延伸任務而言，保守的 Freeze 策略反而略為穩定。

### 6.3 R² 仍為負值，不能宣稱完整正向遷移成功

雖然 MAE 與 RMSE 大幅下降，但 TL 組的 R² 仍約為 -4.95。這表示模型雖然降低了絕對誤差，但尚未完全解釋測試區間的變異趨勢。

因此，Stage B 不應寫成「完整正向遷移成功」，而應保守定位為：

> **Stage B 呈現部分正向遷移效果：Transfer Learning 可大幅降低預測誤差，但模型仍未完整掌握 FwuSow 0–31 後期日齡變異。**

---

## 7. 與 Stage C-1 的比較意義

Stage C-1 為直接遷移對照組：

```text
Kaggle 0–21 → FwuSow 0–31
```

Stage B 則為分階段遷移：

```text
Kaggle 0–21 → FwuSow 0–21 → FwuSow 0–31
```

在相同 `reuse0_21_scaler` 條件下，Stage B 的表現優於 Stage C-1：

| 實驗 | MAE (g) | RMSE (g) | R² |
|---|---:|---:|---:|
| Stage B TL（Freeze） | **391.468** | **453.150** | **-4.954** |
| Stage C-1 TL（Freeze） | 約 790 | 約 832 | 約 -19.07 |

此結果支持：

> **先經過 FwuSow 0–21 中介 fine-tuning，再延伸至 FwuSow 0–31，比直接從 Kaggle 0–21 遷移至 FwuSow 0–31 更穩定。**

---

## 8. Positive Transfer Learning 判斷

Stage B 可判定為：

> **Partial Positive Transfer Learning / 部分正向遷移**

理由如下：

1. 相較 Without TL，Transfer Learning 明顯降低 MAE、MSE 與 RMSE。
2. Freeze 與 Partial FT 均呈現類似改善，代表改善並非單一策略偶然造成。
3. 但 R² 仍為負值，表示模型尚未完整掌握測試區間變異。
4. 因此只能稱為「誤差層面正向改善」，不能稱為「完整正向遷移成功」。

---

## 9. 研究限制

### 9.1 測試視窗數量過少

FwuSow 0–31 的 test set 經 `period = 3` sliding window 後，實際可用測試視窗數很少。因此 R² 對少量測試點非常敏感，結果穩定性有限。

### 9.2 22–31 日齡屬於後期延伸區間

Stage A 權重主要來自 0–21 日齡的對齊任務，而 Stage B 測試區間包含 FwuSow 22–31 日齡後期資料。此區間超出 Kaggle Source 0–21 的主要涵蓋範圍，因此具有外推性質。

### 9.3 `reuse0_21_scaler` 具有研究假設

Stage B 使用 FwuSow 0–21 scaler transform FwuSow 0–31，目的是維持 Stage A 權重尺度一致性。但此策略也會使 22–31 日齡資料可能超出 0–21 normalized range，因此需在論文中說明此設計目的與限制。

### 9.4 單一 seed 與單一切分

目前結果主要基於 seed=1234 與固定時間切分。若要提高統計穩定性，後續可加入多 seed 或 rolling validation，但本階段可作為 LSTM baseline feasibility study。

---

## 10. 論文撰寫建議語句

可用於第四章：

> Stage B 旨在檢驗 Stage A 所得模型權重是否能進一步延伸至 FwuSow 0–31 日齡預測任務。結果顯示，Without TL 組在原始尺度下 MAE 為 2,069.960 g、RMSE 為 2,076.309 g，R² 為 -123.996，顯示從零訓練在後期日齡延伸任務中相當不穩定。相較之下，TL（Freeze）與 TL（Partial FT）皆大幅降低預測誤差，其中 TL（Freeze）取得最低 MAE 391.468 g 與 RMSE 453.150 g。然而，由於兩組 TL 的 R² 仍為負值，表示模型尚未完整掌握 22–31 日齡測試區間之變異。因此，本研究將 Stage B 判定為部分正向遷移，即遷移學習在誤差層面具有明顯改善，但仍未能完整解釋後期生長趨勢。

---

## 11. 建議結論

Stage B 的正式結論建議寫為：

> **Stage B 結果顯示，將 Stage A 權重延伸至 FwuSow 0–31 任務可大幅降低預測誤差，且優於直接遷移對照組 Stage C-1，初步支持分階段遷移策略的有效性。然而，由於 R² 仍為負值，模型尚未完整掌握後期日齡變異，因此本階段應定位為部分正向遷移與後期延伸任務之限制分析，而非完整正向遷移成功。**

---

## 12. 檔案用途建議

本檔案建議放置於：

```text
reports/Chick Weight Result/StageB_FwuSowScr_0-21_to_FwuSowTgt_0-31/results_summary_StageB.md
```

可作為：

1. Group Meeting 結果整理。
2. 第四章實驗結果與分析草稿依據。
3. Stage B 與 Stage C-1 比較表來源。
4. 後續 xLSTM comparison 的 LSTM baseline 記錄。
