# LSTM Baseline Experiment Protocol

## 1. Purpose

This document records the experimental protocol of the LSTM Transfer Learning baseline for the Chick Weight → FwuSow growth curve prediction task.

The purpose of this baseline is not to treat LSTM as the final model, but to verify whether **age-aligned transfer learning** and **staged transfer learning** are useful before introducing xLSTM for further comparison.

This protocol should be treated as the reference document for future xLSTM comparison experiments.

---

## 2. Research Task

### Source domain

- Dataset: Kaggle Chick Weight Dataset
- Age range: 0–21 days
- Role: source-domain pre-training

### Target domain

- Dataset: FwuSow Poultry Dataset
- Stage A target: 0–21 days
- Stage B / C target: 0–31 days
- Role: real-world target-domain smart farming data

### Prediction target

- Target variable: `mean`
- Meaning: daily mean body weight
- Original unit: gram

### Input features

The model uses daily statistical features as input:

```text
min, max, range, p25, p75, IQR, median, std
```

---

## 3. Data Split Protocol

The raw records are first processed into daily statistics.

The daily statistics split used in the LSTM baseline is:

| Dataset | Age Range | Train | Test |
|---|---:|---:|---:|
| Kaggle Chick Weight | 0–21 | 15 | 7 |
| FwuSow Poultry | 0–21 | 15 | 7 |
| FwuSow Poultry | 0–31 | 22 | 10 |

### Chronological split rule

All daily statistics are split in chronological order by age. No random shuffling is applied.

Validation setting:

- `valid_ratio = 0.3`
- `shuffle = False`
- Validation data is split from the training set, not from the test set.
- The final test set is kept independent and used only for final evaluation.

Approximate effective split before sliding-window expansion:

| Dataset | Train before validation | Train after validation | Validation | Test |
|---|---:|---:|---:|---:|
| FwuSow 0–21 | 15 | 10 | 5 | 7 |
| FwuSow 0–31 | 22 | 15 | 7 | 10 |

> Note: The final number of training samples becomes smaller after applying sliding windows.

---

## 4. Sliding Window Setting

The LSTM baseline uses sliding windows for time-series prediction.

Main setting:

- `period = 3`
- `horizon = 1`

The model uses previous time steps to predict the next daily mean body weight.

---

## 5. Stage Design

### Stage 0: Source Pre-training

Kaggle Chick Weight 0–21 is used for source-domain pre-training.

In the pre-training stage, the original source train and test data are merged before validation splitting. This is because source pre-training is used to learn initial temporal representations rather than to report final target-domain performance.

This merged source validation is used only for selecting or confirming the source pre-trained representation. It is not used as final target-domain testing evidence.

Pre-train validation results should not be directly compared with FwuSow target-domain test results.

---

### Stage A: Kaggle 0–21 → FwuSow 0–21

Purpose:

- To verify whether transfer learning improves target prediction under age-aligned 0–21 conditions.

Target:

- FwuSow 0–21

Scaler:

- 0–21 scaler

Experimental modes:

- Without Transfer Learning
- Transfer Learning with Partial Fine-tuning
- Transfer Learning with Freeze

---

### Stage B: Kaggle 0–21 → FwuSow 0–21 → FwuSow 0–31

Purpose:

- To evaluate whether staged transfer improves the 0–31 extension task.

Initial weight:

- Stage A adapted weight

Target:

- FwuSow 0–31

Scaler:

- `reuse0_21_scaler`

Experimental interpretation:

- Stage B is the main staged-transfer experiment.
- It is used to examine whether adaptation on FwuSow 0–21 helps the extended FwuSow 0–31 task.

---

### Stage C-1: Kaggle 0–21 → FwuSow 0–31

Purpose:

- Direct transfer baseline for comparison with Stage B.

Initial weight:

- Kaggle source pre-trained weight

Target:

- FwuSow 0–31

Scaler:

- `reuse0_21_scaler`

Note:

- Stage B and Stage C-1 are comparable because both use `reuse0_21_scaler`.
- The main comparison for staged transfer should be Stage B vs Stage C-1.

---

### Stage C-2: Kaggle 0–21 → FwuSow 0–31

Purpose:

- Scaler sensitivity analysis.

Initial weight:

- Kaggle source pre-trained weight

Target:

- FwuSow 0–31

Scaler:

- `fit0_31_full_scaler`

Note:

- Stage C-2 is a supplementary scaler analysis.
- It should not directly replace the main comparison between Stage B and Stage C-1.
- Because the scaler setting differs from Stage B and Stage C-1, Stage C-2 should be interpreted carefully.

Caution:

- If `fit0_31_full_scaler` was fitted using the full 0–31 dataset, including the test range, this stage should be treated only as scaler sensitivity analysis and not as the primary generalization evaluation.
- A stricter future setting should use a train-only scaler fitted only on the training split.

---

## 6. Model Architecture

The LSTM baseline model consists of:

1. Input layer
2. TimeDistributed Dense layer
3. LSTM layer 1
4. Batch Normalization
5. LSTM layer 2
6. Batch Normalization
7. Dense output layer

Regularization and initialization:

- L2 regularization is used.
- Glorot uniform initialization is used for kernel weights.
- Orthogonal initialization is used for recurrent weights.
- The output layer uses linear activation.

---

## 7. Transfer Learning Strategy

The LSTM baseline includes three main modes:

| Mode | Description |
|---|---|
| Without Transfer Learning | Train on target data from scratch |
| Freeze | Transfer selected layers and freeze transferred layers; only the output layer remains trainable |
| Partial Fine-tuning | Transfer selected layers and fine-tune selected later layers |

### Current transferred layers

According to the current `model.py`, when a pre-trained model is used, the following layers are transferred:

```text
time_distributed_1
batch_normalization_1
lstm_2
batch_normalization_2
```

The first LSTM layer is intentionally not transferred in the current implementation:

```text
lstm_1 is not transferred
```

### Current trainable layers in Freeze mode

In Freeze mode, transferred layers are frozen and only the final output layer remains trainable.

```text
output layer
```

### Current trainable layers in Partial Fine-tuning

In the current Partial Fine-tuning setting, the following layers remain trainable:

```text
lstm_2
output layer
```

This setting is used to reduce overfitting risk under small target-data conditions while still allowing the later temporal representation and output mapping to adapt to the target domain.

---

## 8. Evaluation Metrics

The following metrics are reported:

- MAE
- MSE
- RMSE
- R²

Both normalized-scale and original-scale metrics may be recorded.

The main thesis-reported metrics should be **original-scale metrics after inverse transform**.

Original-scale unit:

- gram

---

## 9. Inverse Transform

The model predictions and ground truth values are transformed back to the original body weight scale before final evaluation.

The main reported MAE, RMSE, and MSE are therefore expressed in gram-based units.

This is important because normalized-scale metrics alone are not directly interpretable in practical body-weight prediction.

---

## 10. Reproducibility Settings

Main random seed:

- `seed = 1234`

Other settings:

- `valid_ratio = 0.3`
- `shuffle = False`
- `period = 3`
- `horizon = 1`
- Test data is only used for final evaluation.

Important limitation:

- The current LSTM baseline mainly reports single-seed results.
- Future xLSTM comparison should preferably include multiple random seeds and report mean ± standard deviation.

---

## 11. Interpretation Rules

### Positive transfer

Transfer learning is considered beneficial when it improves target-domain prediction compared with the without-transfer-learning baseline.

Main indicators:

- Lower MAE
- Lower MSE
- Lower RMSE
- Higher R²

### Strong positive transfer

A result may be interpreted as strong positive transfer when:

- MAE, MSE, and RMSE clearly decrease, and
- R² improves substantially and becomes positive.

### Partial positive transfer

If MAE, MSE, and RMSE improve but R² remains negative, the result should be interpreted as **partial positive transfer**.

This means:

- Transfer learning reduces absolute prediction error.
- However, the model still does not fully explain the target-domain test variance.
- The result should not be over-claimed as solving the 0–31 extension task.

---

## 12. Notes for xLSTM Comparison

Future xLSTM experiments should follow the same protocol as the LSTM baseline:

For fair comparison, xLSTM experiments should not change the train / validation / test split, scaler setting, input features, target variable, or inverse-transform evaluation procedure unless explicitly marked as an additional ablation study.

1. Use the same train / validation / test split.
2. Use `valid_ratio = 0.3`.
3. Use `shuffle = False`.
4. Use the same input features.
5. Use the same target variable.
6. Use the same scaler setting in each Stage.
7. Use inverse transform before reporting original-scale metrics.
8. Compare Stage B and Stage C-1 as the main staged-transfer comparison.
9. Treat Stage C-2 only as scaler sensitivity analysis.
10. Avoid over-claiming when R² remains negative.
11. Record model parameter count, training time, and inference time for fair LSTM vs xLSTM comparison.

---

## 13. Research Positioning

This LSTM baseline is positioned as a feasibility study and comparison baseline for the later xLSTM experiment.

It supports the following research logic:

1. Establish whether transfer learning is useful under age-aligned 0–21 conditions.
2. Examine whether staged transfer is more stable than direct transfer for 0–31 extension.
3. Identify the limitation of late-age extrapolation, especially when R² remains negative.
4. Provide a fair baseline for subsequent xLSTM experiments.

---

## 14. Output Records

Each post-processing output directory should include the following files after running `generate_original_metrics_from_model.py` or `run_inverse_all.sh`:

```text
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
log.txt
```

Additional files may also be generated when using the batch post-processing script:

```text
inverse_run_log.txt
inverse_output_file_list.txt
```

File interpretation:

```text
log.txt = normalized-scale metrics
metrics_original_scale.txt = inverse-transformed original-scale metrics
prediction_compare.csv = normalized and original-scale prediction comparison
```

