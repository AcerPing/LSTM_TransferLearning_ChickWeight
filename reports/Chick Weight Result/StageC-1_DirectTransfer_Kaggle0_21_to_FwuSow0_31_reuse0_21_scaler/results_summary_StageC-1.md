# results_summary_StageC-1.md

## 1. File Position

This file summarizes the LSTM Transfer Learning results for:

```text
StageC-1_DirectTransfer_Kaggle0_21_to_FwuSow0_31_reuse0_21_scaler
```

This stage evaluates the direct transfer setting:

```text
Kaggle Chick Weight 0–21
→ FwuSow Poultry 0–31
```

The target dataset uses:

```text
FwuSow_Poultry_0_31_reuse0_21_scaler
```

The goal is to evaluate whether Kaggle 0–21 pre-trained LSTM weights can improve prediction on FwuSow 0–31 compared with training the target model from scratch.

---

## 2. Stage C-1 Research Role

Stage C-1 is not the main positive-transfer evidence of this research. Instead, it is a **direct transfer comparison group** used to examine whether direct transfer is less stable than staged transfer.

The main comparison logic is:

```text
Stage B:
Kaggle 0–21 → FwuSow 0–21 → FwuSow 0–31

versus

Stage C-1:
Kaggle 0–21 → FwuSow 0–31
```

Because Stage B and Stage C-1 both use `FwuSow_Poultry_0_31_reuse0_21_scaler`, the two stages are comparable under the same target data and scaler setting.

---

## 3. Experiment Design

| Item | Setting |
|---|---|
| Model | LSTM |
| Source dataset | Kaggle Chick Weight Dataset |
| Source age range | 0–21 days |
| Target dataset | FwuSow Poultry Dataset |
| Target age range | 0–31 days |
| Target folder | `FwuSow_Poultry_0_31_reuse0_21_scaler` |
| Scaler setting | Reuse 0–21 scaler |
| Input window | `period = 3` |
| Forecast horizon | 1 step |
| Seed | 1234 |
| Validation ratio | 0.3 |
| Evaluation mode | test |
| Main reported scale | Original scale after inverse transform |

---

## 4. Experiment Groups

### 4.1 Without Transfer Learning

```text
without-transfer-learning/
FwuSow_Poultry_0_31_reuse0_21_scaler/
```

This group trains the LSTM directly on the FwuSow 0–31 target task without using Kaggle pre-trained weights.

### 4.2 Transfer Learning (Partial FT / Unfreeze)

```text
transfer-learning (Unfreeze)/
FwuSow_Poultry_0_31_reuse0_21_scaler/
Kaggle Weight vs Age of Chicks/
```

This group loads Kaggle 0–21 pre-trained weights and performs partial fine-tuning on FwuSow 0–31.

### 4.3 Transfer Learning (Freeze)

```text
transfer-learning (Freeze)/
FwuSow_Poultry_0_31_reuse0_21_scaler/
Kaggle Weight vs Age of Chicks/
```

This group loads Kaggle 0–21 pre-trained weights but freezes transferred layers. Only the final output adaptation is trainable.

---

## 5. Normalized-Scale Metrics

| Mode | MAE | MSE | RMSE | R² | Source |
|---|---:|---:|---:|---:|---|
| Without TL | 1.955371 | 3.846966 | 1.961368 | -123.996222 | `log.txt` |
| TL (Partial FT) | 0.746769 | 0.617763 | 0.785979 | -19.072442 | `params.json` |
| TL (Freeze) | 0.746768 | 0.617761 | 0.785978 | -19.072376 | `params.json` |

The normalized-scale metrics are used only for training-space comparison. The thesis result tables should primarily report original-scale metrics.

---

## 6. Original-Scale Metrics

| Mode | MAE_original (g) | MSE_original (g²) | RMSE_original (g) | R²_original | Source |
|---|---:|---:|---:|---:|---|
| Without TL | 2069.963792 | 4,311,074.816233 | 2076.312793 | -123.996223 | `metrics_original_scale.txt` |
| TL (Partial FT) | 790.534386 | 692,293.289626 | 832.041639 | -19.072499 | `metrics_original_scale.txt` / `original_metrics_params.json` |
| TL (Freeze) | 790.532790 | 692,290.867162 | 832.040184 | -19.072429 | `metrics_original_scale.txt` / `original_metrics_params.json` |

---

## 7. Result Interpretation

### 7.1 Without TL Baseline

The without-transfer-learning baseline performs poorly on the FwuSow 0–31 reuse0_21_scaler task.

The original-scale MAE is greater than 2000 g, RMSE is greater than 2000 g, and R² is highly negative. This indicates that, under this setting, training the LSTM from scratch is unable to stably model the 22–31 day late-age region.

### 7.2 Transfer Learning Results

Both TL (Partial FT) and TL (Freeze) substantially reduce MAE and RMSE compared with Without TL.

The improvement is clear in the error metrics:

```text
Without TL RMSE: about 2076 g
TL RMSE: about 832 g
```

However, R² remains negative in both transfer-learning settings. Therefore, the model still does not adequately explain the variation in the target test set.

### 7.3 Freeze vs Partial FT

The Freeze and Partial FT results are almost identical.

| Comparison | Observation |
|---|---|
| MAE | Freeze is only slightly lower |
| RMSE | Freeze is only slightly lower |
| R² | Freeze is only slightly less negative |
| Practical difference | Negligible |

This suggests that, in Stage C-1, the main limitation is not simply the number of trainable layers. The stronger limitation is likely caused by the direct transfer from a 0–21 source range to a 0–31 target range, where the 22–31 day target range exceeds the source age coverage and the 0–21 scaling range.

---

## 8. Transfer Learning Judgment

| Criterion | Stage C-1 Judgment |
|---|---|
| MAE decreases compared with Without TL | Yes |
| MSE decreases compared with Without TL | Yes |
| RMSE decreases compared with Without TL | Yes |
| R² improves compared with Without TL | Yes |
| R² becomes positive | No |
| Prediction fully captures late-age target trend | No |

Final judgment:

```text
Partial positive transfer / limited transfer effect
```

Stage C-1 supports the claim that direct transfer has some benefit, but it does not support a strong claim of complete positive transfer.

---

## 9. Comparison Meaning with Stage B

Stage C-1 is important because it provides the direct-transfer comparison for Stage B.

If Stage B performs better than Stage C-1 under the same `reuse0_21_scaler` target setting, the result supports the following interpretation:

> Staged transfer through FwuSow 0–21 provides a more stable adaptation path than directly transferring Kaggle 0–21 weights to FwuSow 0–31.

Therefore, the value of Stage C-1 is not that it produces the best model. Its value is that it shows the limitation of direct transfer and supports the need for staged age-aligned adaptation.

---

## 10. Thesis-Safe Interpretation

Recommended wording:

> In Stage C-1, the Kaggle 0–21 pre-trained LSTM was directly transferred to the FwuSow 0–31 target task using the reuse0_21_scaler setting. Compared with the without-transfer-learning baseline, both Freeze and Partial Fine-tuning substantially reduced MAE and RMSE. However, the R² values remained negative, and the model still showed limited ability to capture the late-age 22–31 day target variation. Therefore, Stage C-1 is interpreted as partial positive transfer rather than complete transfer success.

Avoid wording:

```text
Stage C-1 successfully solved FwuSow 0–31 prediction.
The direct transfer model fully captured the 0–31 growth trend.
Stage C-1 proves strong positive transfer.
```

---

## 11. Suggested Thesis Chapter 3 Text

Stage C-1 was designed as a direct transfer comparison experiment. In this setting, the LSTM model pre-trained on Kaggle Chick Weight 0–21 day data was transferred directly to the FwuSow Poultry 0–31 day target task. The target dataset used the `FwuSow_Poultry_0_31_reuse0_21_scaler` setting, in which the 0–31 target data were transformed using the scaler fitted from the 0–21 target range. This setting allows the experiment to examine whether a source model trained on early growth information can directly support an extended target-domain prediction task.

Three experimental modes were compared: training from scratch without transfer learning, transfer learning with frozen transferred layers, and transfer learning with partial fine-tuning. The without-transfer-learning mode served as the target-domain baseline, while the two transfer-learning modes evaluated whether Kaggle pre-trained weights could reduce the prediction error on the FwuSow 0–31 task.

---

## 12. Suggested Thesis Chapter 4 Text

The Stage C-1 results showed that direct transfer from Kaggle 0–21 to FwuSow 0–31 improved the error metrics compared with training from scratch. The without-transfer-learning baseline obtained an original-scale MAE of 2069.964 g and RMSE of 2076.313 g, with R² = -123.996. After applying transfer learning, the Freeze strategy reduced MAE to 790.533 g and RMSE to 832.040 g, while Partial Fine-tuning produced a similar result with MAE = 790.534 g and RMSE = 832.042 g.

Although transfer learning clearly reduced the magnitude of error, both transfer-learning strategies still produced negative R² values. Therefore, Stage C-1 should be interpreted conservatively. The results indicate that Kaggle pre-trained weights can provide partial benefit for the FwuSow 0–31 task, but direct transfer alone is insufficient to fully capture the late-age 22–31 day target growth behavior. This supports the need to compare Stage C-1 with the staged transfer setting in Stage B.

---

## 13. Final Conclusion

Stage C-1 can be summarized as follows:

```text
Direct transfer is better than training from scratch,
but worse than the expected staged-transfer strategy.
```

Recommended conclusion:

> Stage C-1 provides evidence of partial positive transfer. It demonstrates that Kaggle 0–21 pre-trained weights reduce prediction error on the FwuSow 0–31 target task, but the remaining negative R² and late-age underestimation show that direct transfer is limited. The result supports the research value of staged age-aligned transfer learning.
