# Stage C-1 Direct Transfer Experiment

## 1. Folder Scope

This folder records the LSTM Transfer Learning experiment:

```text
StageC-1_DirectTransfer_Kaggle0_21_to_FwuSow0_31_reuse0_21_scaler
```

The experiment evaluates whether a model pre-trained on the Kaggle Chick Weight 0–21 age-range data can be directly transferred to the FwuSow Poultry 0–31 age-range target task under the `reuse0_21_scaler` setting.

This stage is a **direct transfer comparison experiment**, not the main proof of complete positive transfer. Its main role is to compare direct transfer with the staged transfer design used in Stage B.

---

## 2. Research Position

Stage C-1 is designed to answer the following question:

> If the model is transferred directly from Kaggle 0–21 to FwuSow 0–31 without first adapting to FwuSow 0–21, can transfer learning still reduce the target prediction error?

Therefore, Stage C-1 should be interpreted as:

```text
Kaggle 0–21
→ FwuSow 0–31
Direct Transfer
```

It should be compared with:

```text
Stage B:
Kaggle 0–21
→ FwuSow 0–21
→ FwuSow 0–31
Staged Transfer
```

Because both Stage B and Stage C-1 use the `reuse0_21_scaler` setting, they are suitable for comparing staged transfer and direct transfer under the same target scaling condition.

---

## 3. Dataset and Scaler Setting

| Item | Setting |
|---|---|
| Source dataset | Kaggle Chick Weight Dataset |
| Source age range | 0–21 days |
| Target dataset | FwuSow Poultry Dataset |
| Target age range | 0–31 days |
| Target folder | `FwuSow_Poultry_0_31_reuse0_21_scaler` |
| Scaler strategy | Reuse FwuSow 0–21 scaler |
| Input window | `period = 3` |
| Evaluation mode | `test` |
| Main reported scale | Original scale after inverse transform |

The `reuse0_21_scaler` setting means that the FwuSow 0–31 target task is transformed using the scaler fitted from the 0–21 range. This setting intentionally preserves the 0–21 scaling space, but it also makes the 22–31 age range an extrapolation-like region.

---

## 4. Experiment Groups

Stage C-1 contains three main experiment groups.

| Group | Folder | Description |
|---|---|---|
| Without Transfer Learning | `without-transfer-learning/FwuSow_Poultry_0_31_reuse0_21_scaler` | Trains the LSTM from scratch on the FwuSow 0–31 target task |
| Transfer Learning (Partial FT / Unfreeze) | `transfer-learning (Unfreeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/Kaggle Weight vs Age of Chicks` | Uses Kaggle pre-trained weights and performs partial fine-tuning |
| Transfer Learning (Freeze) | `transfer-learning (Freeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/Kaggle Weight vs Age of Chicks` | Uses Kaggle pre-trained weights and freezes transferred layers |

> Note: In thesis writing and presentation, `Unfreeze` should be described as **TL (Partial Fine-tuning / Partial FT)** if the implementation does not fine-tune all model layers.

---

## 5. Main Output Files

Each experiment folder may contain the following files:

| File | Purpose |
|---|---|
| `params.json` | Training configuration and normalized-scale metrics |
| `log.txt` | Normalized-scale metrics and model summary |
| `metrics_original_scale.txt` | Original-scale MAE, MSE, RMSE, and R² |
| `original_metrics_params.json` | Inverse-transform evaluation configuration and original-scale metrics |
| `prediction_compare.csv` | Pairwise prediction comparison data |
| `epoch_log.csv` | Training and validation loss by epoch |
| `inverse_run_log.txt` | Log from inverse-transform metric generation |
| `inverse_output_file_list.txt` | File inventory after inverse-transform output generation |
| `prediction_normalized_scale.png` | Normalized-scale prediction plot |
| `prediction_original_scale.png` | Original-scale prediction plot |
| `yy_plot_normalized_scale.png` | Normalized-scale observed vs predicted plot |
| `yy_plot_original_scale.png` | Original-scale observed vs predicted plot |
| `residual_plot_normalized_scale.png` | Normalized-scale residual plot |
| `residual_plot_original_scale.png` | Original-scale residual plot |
| `error_histogram_normalized_scale.png` | Normalized-scale error histogram |
| `error_histogram_original_scale.png` | Original-scale error histogram |
| `architecture.png` | LSTM architecture diagram |
| `best_model.hdf5` / `*_transferred_best_model.hdf5` | Saved model file |

---

## 6. Original-Scale Result Summary

| Mode | MAE (g) | MSE (g²) | RMSE (g) | R² | Interpretation |
|---|---:|---:|---:|---:|---|
| Without TL | 2069.964 | 4,311,074.816 | 2076.313 | -123.996 | From-scratch baseline fails severely |
| TL (Partial FT) | 790.534 | 692,293.290 | 832.042 | -19.072 | Error reduced, but R² remains negative |
| TL (Freeze) | 790.533 | 692,290.867 | 832.040 | -19.072 | Best but nearly identical to Partial FT |

---

## 7. Interpretation

The Stage C-1 results show that transfer learning improves the target task compared with the without-transfer-learning baseline. Both TL (Freeze) and TL (Partial FT) substantially reduce MAE and RMSE relative to training from scratch.

However, the R² values remain negative, and the prediction behavior still indicates systematic underestimation of the FwuSow 0–31 late-age range. Therefore, Stage C-1 should not be described as complete positive transfer.

The recommended interpretation is:

> Stage C-1 provides evidence of **partial positive transfer** or **limited transfer effect**. Direct transfer from Kaggle 0–21 to FwuSow 0–31 helps reduce prediction error, but it is insufficient for fully modeling the 22–31 age-range extension.

---

## 8. Thesis-Writing Caution

Avoid writing:

```text
Stage C-1 successfully predicts FwuSow 0–31.
Direct transfer fully solves the target task.
Transfer learning completely captures late-age growth.
```

Prefer writing:

```text
Stage C-1 shows that direct transfer can reduce prediction error compared with training from scratch. However, because R² remains negative and the model still underestimates late-age weights, the result is interpreted as partial positive transfer rather than complete transfer success.
```

---

## 9. Recommended Use

This folder can be used for:

1. Direct transfer comparison against Stage B.
2. Supporting the claim that staged transfer is more stable than direct transfer.
3. Discussing the limitation of transferring from a 0–21 source task to a 0–31 target task.
4. Providing practical evidence for the difficulty of late-age extrapolation under the `reuse0_21_scaler` setting.

It should not be used as the main proof of strong positive transfer. The main positive transfer evidence remains Stage A: Kaggle 0–21 → FwuSow 0–21.
