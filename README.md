# LSTM Transfer Learning for Chick Weight Growth Prediction

## Description

This repository contains LSTM-based transfer learning experiments for time-series regression, focusing on chick weight growth prediction.

The current experiment investigates whether a model pre-trained on the public **Kaggle Chick Weight Dataset** can transfer useful temporal growth representations to the **FwuSow Poultry Dataset**, which is used as a real-world smart farming case study.

The main research goal is to evaluate whether transfer learning can improve prediction performance on the target dataset compared with training the model from scratch.

This repository currently serves as the **LSTM baseline and feasibility study** for a master's thesis project. The next research step is to use this baseline as a fair comparison reference for future xLSTM experiments.

---

## Experiment Protocol

The detailed experimental protocol is documented in:

- [experiment_protocol.md](./experiment_protocol.md)

This protocol records:

- Data split rules
- Validation ratio
- Stage A / B / C experiment design
- Scaler settings
- Transfer learning strategy
- Evaluation metrics
- Inverse transform rules
- Notes for future xLSTM comparison

Please read `experiment_protocol.md` before comparing future xLSTM results with this LSTM baseline.

---

## Research Task

### Source Data

- **Kaggle Chick Weight Dataset**
- Age range: 0–21 days
- Purpose: source-domain pre-training

### Target Data

- **FwuSow Poultry Dataset**
- Age range:
  - 0–21 days for age-aligned transfer learning
  - 0–31 days for extended target-domain experiments
- Purpose: real-world smart farming case study

### Prediction Target

The model predicts the next-day chick weight growth trend using daily statistical features.

Input features include:

```text
min, max, range, p25, p75, IQR, median, std
```

The prediction target is:

```text
mean
```

Original unit after inverse transform:

```text
gram
```

---

## Key Experiment Settings

| Item | Setting |
|---|---|
| Model | LSTM |
| Main task | Chick Weight → FwuSow growth curve prediction |
| Input window | `period = 3` |
| Forecast horizon | `horizon = 1` |
| Validation ratio | `valid_ratio = 0.3` |
| Data shuffle | `shuffle = False` |
| Main seed | `seed = 1234` |
| Main metrics | MAE, MSE, RMSE, R² |
| Main reported scale | Original scale after inverse transform |

Important split rule:

```text
X_train / X_test are prepared first.
Then X_valid is split from X_train using valid_ratio = 0.3.
X_test is kept independent and used only for final evaluation.
```

---

## Project Structure

```text
.
├── README.md
├── experiment_protocol.md
├── main.py
├── generate_original_metrics_from_model.py
├── run_inverse_all.sh
├── dataset/
│   ├── source/
│   │   └── Kaggle Weight vs Age of Chicks/
│   │       ├── X_train.pkl
│   │       ├── y_train.pkl
│   │       ├── X_test.pkl
│   │       └── y_test.pkl
│   └── target/
│       ├── FwuSow_Poultry_0~21/
│       ├── FwuSow_Poultry_0_31_reuse0_21_scaler/
│       └── FwuSow_Poultry_0_31_fit0_31_full_scaler/
├── utils/
│   ├── model.py
│   ├── data_io.py
│   ├── save.py
│   └── device.py
├── notebook/
│   └── make_sliding_windows.py
├── reports/
└── archived_scripts/
```

> Note: Actual file availability may vary depending on the local working copy and whether large experiment outputs are tracked by Git.

If `generate_original_metrics_from_model.py` is placed under `notebook/`, run:

```bash
python notebook/generate_original_metrics_from_model.py ...
```

If it is placed in the project root, run:

```bash
python generate_original_metrics_from_model.py ...
```

The current workflow assumes that `generate_original_metrics_from_model.py` is placed in the project root.

---

## Environment Setup

Create and activate the Conda environment:

```bash
conda create -n Fwusow_LSTM_TransferLearning python=3.7
conda activate Fwusow_LSTM_TransferLearning
```

Install required packages according to the project environment.

> Note: The current version of `main.py` disables GPU initialization by default to avoid CUDA / cuDNN compatibility issues.

---

## Supported Training Modes

The current `main.py` supports three main training modes:

| Train Mode | Description |
|---|---|
| `pre-train` | Train the LSTM model on the source dataset |
| `without-transfer-learning` | Train the model from scratch on the target dataset |
| `transfer-learning` | Load a pre-trained model and fine-tune it on the target dataset |

---

## Experiment Workflow

### Stage 0: Kaggle Source Pre-Training

Purpose: train the LSTM model on Kaggle Chick Weight 0–21 days data.

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode pre-train \
  --source-name "Kaggle Weight vs Age of Chicks" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3
```

---

### Stage A Baseline: FwuSow 0–21 Without Transfer Learning

Purpose: establish the target-domain baseline for FwuSow 0–21 days.

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode without-transfer-learning \
  --target-name "FwuSow_Poultry_0~21" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3
```

---

### Stage A: Kaggle 0–21 → FwuSow 0–21 Transfer Learning

Purpose: evaluate whether Kaggle pre-trained weights improve FwuSow 0–21 prediction.

Partial Fine-tuning version:

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode transfer-learning \
  --source-name "Kaggle Weight vs Age of Chicks" \
  --target-name "FwuSow_Poultry_0~21" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3
```

Freeze version:

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode transfer-learning \
  --freeze \
  --source-name "Kaggle Weight vs Age of Chicks" \
  --target-name "FwuSow_Poultry_0~21" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3
```

---

### Stage B Baseline: FwuSow 0–31 Without Transfer Learning

Purpose: establish the baseline for the extended FwuSow 0–31 target task.

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode without-transfer-learning \
  --target-name "FwuSow_Poultry_0_31_reuse0_21_scaler" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3
```

---

### Stage B: Staged Transfer Learning

Purpose: transfer the Stage A adapted model to the FwuSow 0–31 target task.

Scaler:

```text
reuse0_21_scaler
```

Partial Fine-tuning version:

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode transfer-learning \
  --target-name "FwuSow_Poultry_0_31_reuse0_21_scaler" \
  --pre-model-path "reports/Chick Weight Result/pre-train/StageA_Kaggle0_21_to_FwuSow0_21/best_model.hdf5" \
  --source-name "StageA_Kaggle0_21_to_FwuSow0_21" \
  --seed 1234 \
  --nb-epochs 6500 \
  --period 3 \
  --learning-rate 1e-8
```

Freeze version:

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode transfer-learning \
  --freeze \
  --target-name "FwuSow_Poultry_0_31_reuse0_21_scaler" \
  --pre-model-path "reports/Chick Weight Result/pre-train/StageA_Kaggle0_21_to_FwuSow0_21/best_model.hdf5" \
  --source-name "StageA_Kaggle0_21_to_FwuSow0_21" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3 \
  --learning-rate 1e-9
```

---

### Stage C-1: Direct Transfer Learning with reuse0_21_scaler

Purpose: evaluate direct transfer from Kaggle 0–21 to FwuSow 0–31 without the intermediate FwuSow 0–21 adaptation stage.

Scaler:

```text
reuse0_21_scaler
```

Partial Fine-tuning version:

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode transfer-learning \
  --source-name "Kaggle Weight vs Age of Chicks" \
  --target-name "FwuSow_Poultry_0_31_reuse0_21_scaler" \
  --pre-model-path "reports/Chick Weight Result/pre-train/Kaggle Weight vs Age of Chicks/best_model.hdf5" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3 \
  --learning-rate 1e-9
```

Freeze version:

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode transfer-learning \
  --freeze \
  --source-name "Kaggle Weight vs Age of Chicks" \
  --target-name "FwuSow_Poultry_0_31_reuse0_21_scaler" \
  --pre-model-path "reports/Chick Weight Result/pre-train/Kaggle Weight vs Age of Chicks/best_model.hdf5" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3 \
  --learning-rate 1e-9
```

---

### Stage C-2: Direct Transfer Learning with fit0_31_full_scaler

Purpose: conduct supplementary scaler sensitivity analysis.

Scaler:

```text
fit0_31_full_scaler
```

Partial Fine-tuning version:

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode transfer-learning \
  --source-name "Kaggle Weight vs Age of Chicks" \
  --target-name "FwuSow_Poultry_0_31_fit0_31_full_scaler" \
  --pre-model-path "reports/Chick Weight Result/pre-train/Kaggle Weight vs Age of Chicks/best_model.hdf5" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3 \
  --learning-rate 1e-5
```

Freeze version:

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode transfer-learning \
  --freeze \
  --source-name "Kaggle Weight vs Age of Chicks" \
  --target-name "FwuSow_Poultry_0_31_fit0_31_full_scaler" \
  --pre-model-path "reports/Chick Weight Result/pre-train/Kaggle Weight vs Age of Chicks/best_model.hdf5" \
  --seed 1234 \
  --nb-epochs 5000 \
  --period 3 \
  --learning-rate 5e-3
```

> Stage C-2 is a supplementary scaler analysis. It should not directly replace the main comparison between Stage B and Stage C-1.

> Caution: If `fit0_31_full_scaler` was fitted using the full 0–31 dataset, including the test range, Stage C-2 should be treated only as scaler sensitivity analysis and not as the primary generalization evaluation. A stricter future setting should use a train-only scaler fitted only on the training split.

---

## Post-Processing and Inverse Transform

Training produces normalized-scale predictions and metrics. For thesis reporting, predictions should be converted back to the original body weight scale using inverse transform.

Use:

```bash
python generate_original_metrics_from_model.py \
  --dataset-path "dataset/target/FwuSow_Poultry_0~21" \
  --model-path "reports/Chick Weight Result/<stage>/<mode>/<dataset>/best_model.hdf5" \
  --out-dir "reports/Chick Weight Result/<stage>/<mode>/<dataset>" \
  --period 3 \
  --eval-mode test
```

For source pre-training validation:

```bash
python generate_original_metrics_from_model.py \
  --dataset-path "dataset/source/Kaggle Weight vs Age of Chicks" \
  --model-path "reports/Chick Weight Result/<stage>/pre-train/Kaggle Weight vs Age of Chicks/best_model.hdf5" \
  --out-dir "reports/Chick Weight Result/<stage>/pre-train/Kaggle Weight vs Age of Chicks" \
  --period 3 \
  --eval-mode pretrain-valid \
  --valid-ratio 0.3
```

For batch post-processing only, `run_inverse_all.sh` may be used to regenerate inverse-transformed metrics and plots from existing `.hdf5` models.

```bash
bash run_inverse_all.sh
```

`run_inverse_all.sh` is for post-processing existing trained models only. It does not retrain models.

---

## Output Files

Each experiment produces output files under:

```text
reports/Chick Weight Result/
```

Training experiments usually produce:

```text
best_model.hdf5
transferred_best_model.hdf5
epoch_log.csv
params.json
log.txt
Learning Curve.png
architecture.png
```

After running `generate_original_metrics_from_model.py`, the output directory may include:

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
```

If `run_inverse_all.sh` is used, each output directory may additionally include:

```text
inverse_run_log.txt
inverse_output_file_list.txt
```

Notes:

```text
log.txt = normalized-scale metrics
metrics_original_scale.txt = inverse-transformed original-scale metrics
prediction_compare.csv = normalized and original-scale prediction comparison
```

---

## Evaluation Metrics

The following metrics are used to evaluate model performance:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

Final thesis-oriented interpretation should mainly use original-scale metrics after inverse transform.

---

## Positive Transfer Learning Criterion

Transfer learning is considered beneficial when the transfer learning model outperforms the without-transfer-learning baseline on the target dataset.

In this project, positive transfer is evaluated using:

```text
Lower MAE
Lower MSE
Lower RMSE
Higher R²
```

If MAE, MSE, and RMSE improve but R² remains negative, the result is interpreted as **partial positive transfer**, indicating that transfer learning reduces prediction error but does not fully explain the target-domain variance.

---

## Important Interpretation Notes

- Stage A evaluates age-aligned transfer learning under the shared 0–21 day range.
- Stage B evaluates staged transfer from Kaggle 0–21 → FwuSow 0–21 → FwuSow 0–31.
- Stage C-1 evaluates direct transfer from Kaggle 0–21 → FwuSow 0–31 using the same `reuse0_21_scaler` as Stage B.
- Stage B and Stage C-1 are the main comparison for evaluating staged transfer.
- Stage C-2 uses `fit0_31_full_scaler` and should be treated only as supplementary scaler sensitivity analysis.
- If the Stage C-2 scaler was fitted using full 0–31 data, including the test range, Stage C-2 should not be used as the primary evidence for generalization.
- Negative R² should not be over-interpreted. When MAE/RMSE improve but R² remains negative, the result should be described as partial positive transfer rather than complete task success.

---

## Known Limitations

- The current LSTM baseline mainly reports single-seed results.
- Stage C-2 uses `fit0_31_full_scaler` and should be interpreted as scaler sensitivity analysis.
- If the scaler was fitted using the full 0–31 dataset, Stage C-2 should not be used as primary generalization evidence.
- FwuSow 0–31 remains a challenging late-age extension task because R² may remain negative even when MAE/RMSE improve.

---

## Notes

- `train_all.sh`, `transfer_learning.sh`, and older batch training scripts are no longer used in the current workflow.
- Training experiments should be executed using explicit `--source-name`, `--target-name`, and `--pre-model-path` settings.
- For post-processing only, `run_inverse_all.sh` may be used to regenerate inverse-transformed metrics and plots from existing `.hdf5` models.
- Future xLSTM experiments should follow the same split, scaler, target, inverse-transform, and evaluation protocol for fair comparison.

---

## Author

Ho Che Ping

This project is part of a master’s thesis experiment on transfer learning for AI-driven animal production and smart farming time-series prediction.
