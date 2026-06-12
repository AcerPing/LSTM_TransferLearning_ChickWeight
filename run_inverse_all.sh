#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="python"
SCRIPT_PATH="generate_original_metrics_from_model.py"

echo "============================================================"
echo "Run all inverse-transform metrics and plots"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

run_inverse_case () {
  local case_name="$1"
  local dataset_path="$2"
  local model_path="$3"
  local out_dir="$4"
  local eval_mode="$5"
  local valid_ratio="${6:-}"

  echo
  echo "------------------------------------------------------------"
  echo "[START] ${case_name}"
  echo "Dataset : ${dataset_path}"
  echo "Model   : ${model_path}"
  echo "Out dir : ${out_dir}"
  echo "Mode    : ${eval_mode}"
  echo "------------------------------------------------------------"

  if [ ! -d "$dataset_path" ]; then
    echo "[ERROR] Dataset path not found: $dataset_path"
    exit 1
  fi

  if [ ! -f "$model_path" ]; then
    echo "[ERROR] Model file not found: $model_path"
    exit 1
  fi

  mkdir -p "$out_dir"

  if [ "$eval_mode" = "pretrain-valid" ]; then
    $PYTHON_CMD "$SCRIPT_PATH" \
      --dataset-path "$dataset_path" \
      --model-path "$model_path" \
      --out-dir "$out_dir" \
      --period 3 \
      --eval-mode "$eval_mode" \
      --valid-ratio "$valid_ratio" \
      2>&1 | tee "$out_dir/inverse_run_log.txt"
  else
    $PYTHON_CMD "$SCRIPT_PATH" \
      --dataset-path "$dataset_path" \
      --model-path "$model_path" \
      --out-dir "$out_dir" \
      --period 3 \
      --eval-mode "$eval_mode" \
      2>&1 | tee "$out_dir/inverse_run_log.txt"
  fi

  {
    echo "Case name: ${case_name}"
    echo "Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Dataset path: ${dataset_path}"
    echo "Model path: ${model_path}"
    echo "Output directory: ${out_dir}"
    echo
    echo "Files in output directory:"
    find "$out_dir" -maxdepth 1 -type f -printf "%f\n" | sort
  } > "$out_dir/inverse_output_file_list.txt"

  echo "[DONE] ${case_name}"
}

# ============================================================
# Stage A
# ============================================================

run_inverse_case \
  "Inverse 1：Pre-Train / Kaggle Source" \
  "dataset/source/Kaggle Weight vs Age of Chicks" \
  "reports/Chick Weight Result/StageA_KaggleSrc_0-21_to_FwuSowTgt_0-21/pre-train/Kaggle Weight vs Age of Chicks/best_model.hdf5" \
  "reports/Chick Weight Result/StageA_KaggleSrc_0-21_to_FwuSowTgt_0-21/pre-train/Kaggle Weight vs Age of Chicks" \
  "pretrain-valid" \
  "0.3"

run_inverse_case \
  "Inverse 2：Stage A Transfer Learning Unfreeze" \
  "dataset/target/FwuSow_Poultry_0~21" \
  "reports/Chick Weight Result/StageA_KaggleSrc_0-21_to_FwuSowTgt_0-21/transfer-learning (Unfreeze)/FwuSow_Poultry_0~21/Kaggle Weight vs Age of Chicks/Kaggle Weight vs Age of Chicks_transferred_best_model.hdf5" \
  "reports/Chick Weight Result/StageA_KaggleSrc_0-21_to_FwuSowTgt_0-21/transfer-learning (Unfreeze)/FwuSow_Poultry_0~21/Kaggle Weight vs Age of Chicks" \
  "test"

run_inverse_case \
  "Inverse 3：Stage A Transfer Learning Freeze" \
  "dataset/target/FwuSow_Poultry_0~21" \
  "reports/Chick Weight Result/StageA_KaggleSrc_0-21_to_FwuSowTgt_0-21/transfer-learning (Freeze)/FwuSow_Poultry_0~21/Kaggle Weight vs Age of Chicks/Kaggle Weight vs Age of Chicks_transferred_best_model.hdf5" \
  "reports/Chick Weight Result/StageA_KaggleSrc_0-21_to_FwuSowTgt_0-21/transfer-learning (Freeze)/FwuSow_Poultry_0~21/Kaggle Weight vs Age of Chicks" \
  "test"

run_inverse_case \
  "Inverse 4：Stage A Without Transfer Learning" \
  "dataset/target/FwuSow_Poultry_0~21" \
  "reports/Chick Weight Result/StageA_KaggleSrc_0-21_to_FwuSowTgt_0-21/without-transfer-learning/FwuSow_Poultry_0~21/best_model.hdf5" \
  "reports/Chick Weight Result/StageA_KaggleSrc_0-21_to_FwuSowTgt_0-21/without-transfer-learning/FwuSow_Poultry_0~21" \
  "test"

# ============================================================
# Stage B
# ============================================================

run_inverse_case \
  "Inverse 5：Stage B / Stage C-1 Baseline Without TL" \
  "dataset/target/FwuSow_Poultry_0_31_reuse0_21_scaler" \
  "reports/Chick Weight Result/StageB_FwuSowScr_0-21_to_FwuSowTgt_0-31/without-transfer-learning/FwuSow_Poultry_0_31_reuse0_21_scaler/best_model.hdf5" \
  "reports/Chick Weight Result/StageB_FwuSowScr_0-21_to_FwuSowTgt_0-31/without-transfer-learning/FwuSow_Poultry_0_31_reuse0_21_scaler" \
  "test"

run_inverse_case \
  "Inverse 6：Stage B Transfer Learning Unfreeze" \
  "dataset/target/FwuSow_Poultry_0_31_reuse0_21_scaler" \
  "reports/Chick Weight Result/StageB_FwuSowScr_0-21_to_FwuSowTgt_0-31/transfer-learning (Unfreeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/StageA_Kaggle0_21_to_FwuSow0_21/StageA_Kaggle0_21_to_FwuSow0_21_transferred_best_model.hdf5" \
  "reports/Chick Weight Result/StageB_FwuSowScr_0-21_to_FwuSowTgt_0-31/transfer-learning (Unfreeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/StageA_Kaggle0_21_to_FwuSow0_21" \
  "test"

run_inverse_case \
  "Inverse 7：Stage B Transfer Learning Freeze" \
  "dataset/target/FwuSow_Poultry_0_31_reuse0_21_scaler" \
  "reports/Chick Weight Result/StageB_FwuSowScr_0-21_to_FwuSowTgt_0-31/transfer-learning (Freeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/StageA_Kaggle0_21_to_FwuSow0_21/StageA_Kaggle0_21_to_FwuSow0_21_transferred_best_model.hdf5" \
  "reports/Chick Weight Result/StageB_FwuSowScr_0-21_to_FwuSowTgt_0-31/transfer-learning (Freeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/StageA_Kaggle0_21_to_FwuSow0_21" \
  "test"

# ============================================================
# Stage C-1
# ============================================================

run_inverse_case \
  "Inverse 8：Stage C-1 Direct Transfer Unfreeze" \
  "dataset/target/FwuSow_Poultry_0_31_reuse0_21_scaler" \
  "reports/Chick Weight Result/StageC-1_DirectTransfer_Kaggle0_21_to_FwuSow0_31_reuse0_21_scaler/transfer-learning (Unfreeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/Kaggle Weight vs Age of Chicks/Kaggle Weight vs Age of Chicks_transferred_best_model.hdf5" \
  "reports/Chick Weight Result/StageC-1_DirectTransfer_Kaggle0_21_to_FwuSow0_31_reuse0_21_scaler/transfer-learning (Unfreeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/Kaggle Weight vs Age of Chicks" \
  "test"

run_inverse_case \
  "Inverse 9：Stage C-1 Direct Transfer Freeze" \
  "dataset/target/FwuSow_Poultry_0_31_reuse0_21_scaler" \
  "reports/Chick Weight Result/StageC-1_DirectTransfer_Kaggle0_21_to_FwuSow0_31_reuse0_21_scaler/transfer-learning (Freeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/Kaggle Weight vs Age of Chicks/Kaggle Weight vs Age of Chicks_transferred_best_model.hdf5" \
  "reports/Chick Weight Result/StageC-1_DirectTransfer_Kaggle0_21_to_FwuSow0_31_reuse0_21_scaler/transfer-learning (Freeze)/FwuSow_Poultry_0_31_reuse0_21_scaler/Kaggle Weight vs Age of Chicks" \
  "test"

# ============================================================
# Stage C-2
# ============================================================

run_inverse_case \
  "Inverse 10：Stage C-2 Baseline Without TL" \
  "dataset/target/FwuSow_Poultry_0_31_fit0_31_full_scaler" \
  "reports/Chick Weight Result/StageC-2_DirectTransfer_Kaggle0_21_to_FwuSow0_31_fit0_31_full_scaler/without-transfer-learning/FwuSow_Poultry_0_31_fit0_31_full_scaler/best_model.hdf5" \
  "reports/Chick Weight Result/StageC-2_DirectTransfer_Kaggle0_21_to_FwuSow0_31_fit0_31_full_scaler/without-transfer-learning/FwuSow_Poultry_0_31_fit0_31_full_scaler" \
  "test"

run_inverse_case \
  "Inverse 11：Stage C-2 Direct Transfer Unfreeze" \
  "dataset/target/FwuSow_Poultry_0_31_fit0_31_full_scaler" \
  "reports/Chick Weight Result/StageC-2_DirectTransfer_Kaggle0_21_to_FwuSow0_31_fit0_31_full_scaler/transfer-learning (Unfreeze)/FwuSow_Poultry_0_31_fit0_31_full_scaler/Kaggle Weight vs Age of Chicks/Kaggle Weight vs Age of Chicks_transferred_best_model.hdf5" \
  "reports/Chick Weight Result/StageC-2_DirectTransfer_Kaggle0_21_to_FwuSow0_31_fit0_31_full_scaler/transfer-learning (Unfreeze)/FwuSow_Poultry_0_31_fit0_31_full_scaler/Kaggle Weight vs Age of Chicks" \
  "test"

run_inverse_case \
  "Inverse 12：Stage C-2 Direct Transfer Freeze" \
  "dataset/target/FwuSow_Poultry_0_31_fit0_31_full_scaler" \
  "reports/Chick Weight Result/StageC-2_DirectTransfer_Kaggle0_21_to_FwuSow0_31_fit0_31_full_scaler/transfer-learning (Freeze)/FwuSow_Poultry_0_31_fit0_31_full_scaler/Kaggle Weight vs Age of Chicks/Kaggle Weight vs Age of Chicks_transferred_best_model.hdf5" \
  "reports/Chick Weight Result/StageC-2_DirectTransfer_Kaggle0_21_to_FwuSow0_31_fit0_31_full_scaler/transfer-learning (Freeze)/FwuSow_Poultry_0_31_fit0_31_full_scaler/Kaggle Weight vs Age of Chicks" \
  "test"

echo
echo "============================================================"
echo "All inverse-transform tasks completed."
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"