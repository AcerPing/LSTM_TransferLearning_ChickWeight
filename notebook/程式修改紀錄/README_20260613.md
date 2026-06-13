# LSTM Transfer Learning for Chick Weight Growth Prediction

## Description

This repository contains LSTM-based transfer learning experiments for time-series regression, focusing on chick weight growth prediction.

The current experiment investigates whether a model pre-trained on the public **Kaggle Chick Weight Dataset** can transfer useful temporal growth representations to the **FwuSow Poultry Dataset**, which is used as a real-world smart farming case study.

The main research goal is to evaluate whether transfer learning can improve prediction performance on the target dataset compared with training the model from scratch.

---

## Research Task

### Source Data

* **Kaggle Chick Weight Dataset**
* Age range: 0–21 days
* Purpose: source-domain pre-training

### Target Data

* **FwuSow Poultry Dataset**
* Age range:

  * 0–21 days for age-aligned transfer learning
  * 0–31 days for extended target-domain experiments
* Purpose: real-world smart farming case study

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

---

## Project Structure

```text
.
├── main.py
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
└── README.md
```

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

| Train Mode                  | Description                                                     |
| --------------------------- | --------------------------------------------------------------- |
| `pre-train`                 | Train the LSTM model on the source dataset                      |
| `without-transfer-learning` | Train the model from scratch on the target dataset              |
| `transfer-learning`         | Load a pre-trained model and fine-tune it on the target dataset |

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

### Stage A: Kaggle 0–21 to FwuSow 0–21 Transfer Learning

Purpose: evaluate whether Kaggle pre-trained weights improve FwuSow 0–21 prediction.

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

Purpose: transfer the Stage A model to the FwuSow 0–31 target task.

```bash
python main.py --out-dir "Chick Weight Result" \
  --train-mode transfer-learning \
  --target-name "FwuSow_Poultry_0_31_reuse0_21_scaler" \
  --pre-model-path "reports/Chick Weight Result/pre-train/StageA_Kaggle0_21_to_FwuSow0_21/best_model.hdf5" \
  --source-name "StageA_Kaggle0_21_to_FwuSow0_21" \
  --seed 1234 \
  --nb-epochs 6500 \
  --period 3 \
  --learning-rate 1e-9
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

### Stage C: Direct Transfer Learning

Purpose: evaluate direct transfer from Kaggle 0–21 to FwuSow 0–31 without the intermediate FwuSow 0–21 fine-tuning stage.

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

## Output Files

Each experiment produces output files under:

```text
reports/Chick Weight Result/
```

Typical output files include:

```text
best_model.hdf5
epoch_log.csv
params.json
log.txt
prediction.png
yy_plot.png
Residual Plot.png
Error Histogram.png
Learning Curve.png
architecture.png
```

---

## Evaluation Metrics

The following metrics are used to evaluate model performance:

| Metric | Description                  |
| ------ | ---------------------------- |
| MAE    | Mean Absolute Error          |
| MSE    | Mean Squared Error           |
| RMSE   | Root Mean Squared Error      |
| R²     | Coefficient of Determination |

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

## Notes

* `train_all.sh`, `transfer_learning.sh`, and older batch scripts are no longer used in the current workflow.
* Experiments should be executed using explicit `--source-name`, `--target-name`, and `--pre-model-path` settings.
* The current workflow avoids automatic execution scripts to reduce the risk of using incorrect datasets or model weights.
* Stage B and Stage C should be interpreted carefully because different scaler strategies may represent different experimental assumptions.

---

## Author

Ho Che Ping

This project is part of a master’s thesis experiment on transfer learning for AI-driven animal production and smart farming time-series prediction.
