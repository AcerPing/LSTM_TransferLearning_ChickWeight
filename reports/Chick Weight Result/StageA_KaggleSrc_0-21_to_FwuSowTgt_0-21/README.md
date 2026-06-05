# Stage A：Kaggle Source 0–21 → FwuSow Target 0–21

本資料夾保存 Stage A 日齡對齊式遷移學習實驗結果。

## 實驗目的

驗證在 SourceData 與 TargetData 皆限制於 0–21 日齡範圍時，Kaggle Chick Weight SourceData 預訓練權重是否能改善 FwuSow TargetData 雞隻體重預測效果。

## 資料設定

- SourceData：Kaggle Chick Weight 0–21 日齡
- TargetData：FwuSow Poultry 0–21 日齡
- Input features：min, max, range, p25, p75, IQR, median, std
- Prediction target：mean
- Period / Time Window：3

## 實驗組別

1. Pre-train：Kaggle SourceData 預訓練
2. Without Transfer Learning：FwuSow TargetData 從零訓練
3. Transfer Learning (Freeze)：凍結式遷移學習
4. Transfer Learning (Unfreeze)：部分層微調式遷移學習

## 結果摘要

- Without TL：作為 Target baseline
- Freeze TL：相較 Without TL 明顯改善，但 R² 仍略低於 0
- Partial Fine-tuning TL：整體表現最佳，可作為 Stage A 主要正向遷移結果