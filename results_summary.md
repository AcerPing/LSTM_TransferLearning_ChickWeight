# LSTM Baseline Results Summary

## 1. Purpose

This document summarizes the main experimental results of the **LSTM Transfer Learning baseline** for the Chick Weight → FwuSow growth curve prediction task.

This result summary is intended to support:

- thesis Chapter 4 result organization,
- future xLSTM comparison experiments,
- quick review of Stage A / B / C results,
- interpretation of positive transfer learning.

The current LSTM experiment should be interpreted as a **baseline and feasibility study**, not as the final model of the thesis.

---

## 2. Experiment Overview

### Research task

| Item | Description |
|---|---|
| Source dataset | Kaggle Chick Weight Dataset |
| Target dataset | FwuSow Poultry Dataset |
| Source age range | 0–21 days |
| Target age range | 0–21 days and 0–31 days |
| Prediction target | `mean` daily body weight |
| Original unit | gram |
| Input window | `period = 3` |
| Forecast horizon | `horizon = 1` |
| Validation ratio | `valid_ratio = 0.3` |
| Main seed | `seed = 1234` |
| Main reported scale | Original scale after inverse transform |

### Input features

```text
min, max, range, p25, p75, IQR, median, std
```

### Evaluation metrics

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination |

---

## 3. Dataset Summary

| Dataset | Age Range | Records | Growth Curves | Recorded Weight Range | Daily Mean Weight Range | Daily Stats Split |
|---|---:|---:|---:|---:|---:|---|
| Kaggle | 0–21 | 880 | 40 | 39–373 g | 41.075–228.175 g | 15 train / 7 test |
| FwuSow | 0–21 | 1056 | 48 | 20–1465 g | 37.167–1095.771 g | 15 train / 7 test |
| FwuSow | 0–31 | 1536 | 48 | 20–2354 g | 37.167–1998.542 g | 22 train / 10 test |

Notes:

- Dataset records refer to cleaned long-format growth records.
- Model training uses daily statistical features after data aggregation.
- Train / test split is based on daily statistics in chronological order.
- No random shuffling is applied.

---

## 4. Stage Design

| Stage | Experiment | Source / Pre-model | Target | Scaler | Purpose |
|---|---|---|---|---|---|
| Stage 0 | Kaggle Source Pre-train | Kaggle 0–21 | Source validation | Kaggle scaler | Learn early growth representation |
| Stage A | Kaggle 0–21 → FwuSow 0–21 | Kaggle 0–21 | FwuSow 0–21 | 0–21 scaler | Age-aligned transfer learning |
| Stage B | Kaggle 0–21 → FwuSow 0–21 → FwuSow 0–31 | Stage A adapted weights | FwuSow 0–31 | reuse0_21_scaler | Staged transfer learning |
| Stage C-1 | Kaggle 0–21 → FwuSow 0–31 | Kaggle 0–21 | FwuSow 0–31 | reuse0_21_scaler | Direct transfer comparison |
| Stage C-2 | Kaggle 0–21 → FwuSow 0–31 | Kaggle 0–21 | FwuSow 0–31 | fit0_31_full_scaler | Supplementary scaler sensitivity analysis |

---

## 5. Stage 0 Result: Kaggle Source Pre-training

**Experiment:** Kaggle Chick Weight 0–21 source-domain pre-training  
**Evaluation scale:** Original scale after inverse transform

| Mode | MSE (g²) | RMSE (g) | MAE (g) | R² | Interpretation |
|---|---:|---:|---:|---:|---|
| Pre-train | 3.080 | 1.755 | 1.597 | 0.976 | Source model learned the Kaggle early growth trend well |

### Interpretation

The Kaggle source pre-training result shows that the LSTM model can learn the early growth trend in the source domain. Therefore, the model weights can be used as the initial weights for subsequent transfer learning experiments.

---

## 6. Stage A Result: Kaggle 0–21 → FwuSow 0–21

**Experiment:** Age-aligned transfer learning  
**Evaluation scale:** Original scale after inverse transform

| Mode | MSE (g²) | RMSE (g) | MAE (g) | R² | Interpretation |
|---|---:|---:|---:|---:|---|
| Without TL | 171,083.974 | 413.623 | 395.418 | -15.801 | Target from-scratch training was unstable |
| TL Partial FT | **3,050.515** | **55.231** | **50.158** | **0.700** | Best Stage A result; clear positive transfer |
| TL Freeze | 5,088.740 | 71.335 | 59.149 | 0.500 | Clear improvement over Without TL |

### Interpretation

Stage A provides the strongest evidence of positive transfer learning in this LSTM baseline. Under the shared 0–21 day age range, transfer learning substantially reduced MAE and RMSE, and R² improved from a negative value to a positive value.

This suggests that the Kaggle source pre-trained model learned useful early growth representations that can be adapted to the FwuSow 0–21 target task.

---

## 7. Stage B Result: Staged Transfer to FwuSow 0–31

**Experiment:** Kaggle 0–21 → FwuSow 0–21 → FwuSow 0–31  
**Scaler:** reuse0_21_scaler  
**Evaluation scale:** Original scale after inverse transform

| Mode | MSE (g²) | RMSE (g) | MAE (g) | R² | Interpretation |
|---|---:|---:|---:|---:|---|
| Without TL | 4,311,058.519 | 2,076.309 | 2,069.960 | -123.996 | From-scratch training was highly unstable |
| TL Partial FT | 205,469.715 | 453.288 | 391.633 | -4.957 | Large error reduction, but R² remained negative |
| TL Freeze | **205,345.313** | **453.150** | **391.468** | **-4.954** | Best Stage B result; partial positive transfer |

### Interpretation

Stage B shows that transfer learning can greatly reduce MAE and RMSE in the 0–31 extended target task. However, R² remains negative, indicating that the model did not fully explain the variance in the late-age target test range.

Therefore, Stage B should be interpreted as **partial positive transfer** or **error-level improvement**, not as complete success in solving the FwuSow 0–31 prediction task.

---

## 8. Stage C-1 Result: Direct Transfer with reuse0_21_scaler

**Experiment:** Kaggle 0–21 → FwuSow 0–31  
**Scaler:** reuse0_21_scaler  
**Evaluation scale:** Original scale after inverse transform

| Mode | MSE (g²) | RMSE (g) | MAE (g) | R² | Interpretation |
|---|---:|---:|---:|---:|---|
| Without TL | 4,311,074.816 | 2,076.313 | 2,069.964 | -123.996 | From-scratch training was highly unstable |
| TL Partial FT | 692,293.290 | 832.042 | 790.534 | -19.072 | Error reduced, but direct transfer remained limited |
| TL Freeze | **692,290.867** | **832.040** | **790.533** | **-19.072** | Best Stage C-1 result; weaker than Stage B |

### Interpretation

Stage C-1 confirms that direct transfer from Kaggle 0–21 to FwuSow 0–31 can still reduce prediction error compared with Without TL. However, it performs clearly worse than Stage B.

This supports the value of staged transfer learning:

```text
Stage B > Stage C-1 > Without TL
```

In other words, adapting the model first to FwuSow 0–21 before extending to FwuSow 0–31 is more effective than direct transfer to the full 0–31 target task.

---

## 9. Stage C-2 Result: Direct Transfer with fit0_31_full_scaler

**Experiment:** Kaggle 0–21 → FwuSow 0–31  
**Scaler:** fit0_31_full_scaler  
**Evaluation scale:** Original scale after inverse transform  
**Positioning:** Supplementary scaler sensitivity analysis

| Mode | MSE (g²) | RMSE (g) | MAE (g) | R² | Interpretation |
|---|---:|---:|---:|---:|---|
| Without TL | 6,055,704.457 | 2,460.834 | 2,419.827 | -174.580 | From-scratch training was highly unstable |
| TL Partial FT | 333,119.613 | 577.165 | 502.441 | -8.659 | Clear improvement over Without TL |
| TL Freeze | **236,755.496** | **486.575** | **420.612** | **-5.865** | Best Stage C-2 result, but R² remained negative |

### Interpretation

Stage C-2 shows that transfer learning can reduce MAE and RMSE even when FwuSow 0–31 is treated as a complete target task with its own scaler. However, Stage C-2 should be interpreted carefully.

If `fit0_31_full_scaler` was fitted using the full 0–31 dataset, including the test range, this experiment should only be treated as **scaler sensitivity analysis**, not as primary evidence of generalization.

Stage C-2 does not replace the main comparison between Stage B and Stage C-1.

---

## 10. Main Comparison: Stage B vs Stage C-1

The most important comparison for evaluating staged transfer learning is:

| Experiment | Scaler | Best Mode | RMSE (g) | MAE (g) | R² |
|---|---|---|---:|---:|---:|
| Stage B | reuse0_21_scaler | TL Freeze | **453.150** | **391.468** | **-4.954** |
| Stage C-1 | reuse0_21_scaler | TL Freeze | 832.040 | 790.533 | -19.072 |

### Interpretation

Because both Stage B and Stage C-1 use the same FwuSow 0–31 target task and the same `reuse0_21_scaler`, they can be compared more directly.

The results show that Stage B has lower MAE and RMSE than Stage C-1. This supports the interpretation that staged transfer learning is more stable than direct transfer in this task.

---

## 11. Positive Transfer Learning Interpretation

### Suggested interpretation standard

| Case | Interpretation |
|---|---|
| MAE / RMSE improve and R² becomes positive | Clear positive transfer |
| MAE / RMSE improve but R² remains negative | Partial positive transfer |
| MAE / RMSE worsen or R² decreases | Negative transfer |
| Different scaler settings | Interpret as supplementary or sensitivity analysis unless comparison conditions are identical |

### Stage-level interpretation

| Stage | Best Result | Suggested Interpretation |
|---|---|---|
| Stage A | TL Partial FT | Clear positive transfer |
| Stage B | TL Freeze | Partial positive transfer / error-level improvement |
| Stage C-1 | TL Freeze | Partial positive transfer, weaker than Stage B |
| Stage C-2 | TL Freeze | Supplementary scaler sensitivity result |

---

## 12. Overall Conclusion

The LSTM baseline results suggest the following:

1. **Stage A provides clear evidence of positive transfer learning** under the shared 0–21 day age range.
2. **Stage B shows that staged transfer learning can reduce prediction error** in the extended FwuSow 0–31 task.
3. **Stage C-1 confirms that direct transfer is less effective than staged transfer** under the same `reuse0_21_scaler` condition.
4. **Stage C-2 should be treated as supplementary scaler sensitivity analysis**, especially if the scaler was fitted using the full 0–31 dataset.
5. **FwuSow 0–31 remains a difficult late-age extension task**, because R² remains negative even when MAE and RMSE improve.

Therefore, the LSTM baseline supports the research value of age-aligned and staged transfer learning. The next step is to apply the same Stage A / B / C protocol to xLSTM and compare whether xLSTM can further improve the late-age extension task and small-sample transfer learning performance.

---

## 13. Notes for Future xLSTM Comparison

For fair comparison, xLSTM experiments should follow the same protocol:

- same train / validation / test split,
- same input features,
- same target variable,
- same scaler setting for each stage,
- same inverse-transform evaluation rule,
- same metrics,
- same Stage A / B / C structure.

If any of these settings are changed, the experiment should be clearly marked as an additional ablation study rather than a direct LSTM-vs-xLSTM comparison.

---

## 14. Known Limitations

- The current LSTM baseline mainly reports single-seed results.
- Stage C-2 uses `fit0_31_full_scaler` and should be interpreted as scaler sensitivity analysis.
- If the scaler was fitted using the full 0–31 dataset, Stage C-2 should not be used as primary generalization evidence.
- FwuSow 0–31 remains a challenging late-age extension task because R² may remain negative even when MAE/RMSE improve.
- Without TL results are very weak in the 0–31 task; future work may further verify whether this is due to task difficulty, small sample size, or from-scratch training instability.
