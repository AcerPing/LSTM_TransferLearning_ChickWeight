# LSTM Baseline Experiment Protocol

## 1. Purpose

This document records the experimental protocol of the LSTM Transfer Learning baseline for the Chick Weight to FwuSow growth curve prediction task.

The purpose of this baseline is not to treat LSTM as the final model, but to verify whether age-aligned transfer learning and staged transfer strategies are useful before introducing xLSTM for further comparison.

## 2. Research Task

Source domain:

- Kaggle Chick Weight Dataset
- Age range: 0–21 days

Target domain:

- FwuSow Poultry Dataset
- Stage A: 0–21 days
- Stage B / C: 0–31 days

Prediction target:

- Daily mean body weight
- Original unit: gram

## 3. Data Split Protocol

The dataset is first processed into daily statistics.

The daily statistics split used in the LSTM baseline is:

| Dataset | Age Range | Train | Test |
|---|---:|---:|---:|
| Kaggle Chick Weight | 0–21 | 15 | 7 |
| FwuSow Poultry | 0–21 | 15 | 7 |
| FwuSow Poultry | 0–31 | 22 | 10 |

During model training, validation data is further split from the training data.

Validation setting:

- valid_ratio = 0.3
- shuffle = False
- validation data is split from the training set, not from the test set

Therefore, the final evaluation test set is kept independent from validation.

## 4. Sliding Window Setting

The LSTM baseline uses sliding windows for time-series prediction.

Main setting:

- period = 3
- horizon = 1

The model uses previous time steps to predict the next daily mean body weight.

## 5. Stage Design

### Stage 0: Source Pre-training

Kaggle Chick Weight 0–21 is used for source-domain pre-training.

In the pre-training stage, the original source train and test data are merged before splitting validation data. This is because source pre-training is used to learn initial temporal representations rather than to report final target-domain performance.

### Stage A: Kaggle 0–21 to FwuSow 0–21

Purpose:

- To verify whether transfer learning improves target prediction under age-aligned 0–21 conditions.

Target:

- FwuSow 0–21

Scaler:

- 0–21 scaler

### Stage B: Kaggle 0–21 to FwuSow 0–21 to FwuSow 0–31

Purpose:

- To evaluate whether staged transfer improves the 0–31 extension task.

Initial weight:

- Stage A adapted weight

Target:

- FwuSow 0–31

Scaler:

- reuse0_21_scaler

### Stage C-1: Kaggle 0–21 to FwuSow 0–31

Purpose:

- Direct transfer baseline for comparison with Stage B.

Initial weight:

- Kaggle source pre-trained weight

Target:

- FwuSow 0–31

Scaler:

- reuse0_21_scaler

Note:

- Stage B and Stage C-1 are comparable because both use reuse0_21_scaler.

### Stage C-2: Kaggle 0–21 to FwuSow 0–31

Purpose:

- Scaler sensitivity analysis.

Initial weight:

- Kaggle source pre-trained weight

Target:

- FwuSow 0–31

Scaler:

- fit0_31_full_scaler

Note:

- Stage C-2 is a supplementary scaler analysis.
- It should not directly replace the main comparison between Stage B and Stage C-1.

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

## 7. Transfer Learning Strategy

The LSTM baseline includes three main modes:

| Mode | Description |
|---|---|
| Without Transfer Learning | Train on target data from scratch |
| Freeze | Transfer selected layers and freeze transferred layers |
| Partial Fine-tuning | Transfer selected layers and fine-tune selected later layers |

In the current Partial Fine-tuning setting, selected transferred layers are loaded from the pre-trained model, while only later LSTM layers and the output layer are trainable.

## 8. Evaluation Metrics

The following metrics are reported:

- MAE
- MSE
- RMSE
- R²

Both normalized-scale and original-scale metrics may be recorded.

The main thesis-reported metrics should be original-scale metrics after inverse transform.

Original-scale unit:

- gram

## 9. Inverse Transform

The model predictions and ground truth values are transformed back to the original body weight scale before final evaluation.

The main reported MAE, RMSE, and MSE are therefore expressed in gram-based units.

## 10. Reproducibility Settings

Main random seed:

- seed = 1234

Other settings:

- shuffle = False
- validation ratio = 0.3
- test data is only used for final evaluation

## 11. Notes for xLSTM Comparison

The future xLSTM experiments should follow the same protocol as the LSTM baseline:

1. Use the same train / validation / test split.
2. Use valid_ratio = 0.3.
3. Use shuffle = False.
4. Use the same input features.
5. Use the same target variable.
6. Use the same scaler setting in each Stage.
7. Use inverse transform before reporting original-scale metrics.
8. Compare Stage B and Stage C-1 as the main staged-transfer comparison.
9. Treat Stage C-2 only as scaler sensitivity analysis.
10. Avoid over-claiming when R² remains negative.