"""
讀取指定 dataset
讀取指定 best_model.hdf5
用相同 period 建立 sliding windows
預測 y_pred
輸出 pred_vs_true.csv
輸出 prediction plot / yy plot / residual plot / error histogram
輸出 metrics
"""

import argparse
import json
from os import path, makedirs

import numpy as np
import pandas as pd
from keras.models import load_model

from utils.model import rmse
from utils.data_io import read_data_from_dataset
from utils.save import (
    save_prediction_plot,
    save_yy_plot,
    save_mse,
    ResidualPlot,
    ErrorHistogram
)
from notebook.make_sliding_windows import make_sliding_windows


def parse_arguments():
    ap = argparse.ArgumentParser(
        description="Analyze trained LSTM model on a specified dataset."
    )

    ap.add_argument(
        "--dataset-path",
        required=True,
        type=str,
        help="Path to dataset folder containing X_train.pkl, y_train.pkl, X_test.pkl, y_test.pkl"
    )

    ap.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to trained .hdf5 model"
    )

    ap.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Output directory for analysis results"
    )

    ap.add_argument(
        "--period",
        default=3,
        type=int,
        help="Number of time steps used as input window"
    )

    ap.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "test", "all"],
        help="Which split to evaluate: train, test, or all"
    )

    ap.add_argument(
        "--feature-names",
        nargs="*",
        default=["min", "max", "range", "p25", "p75", "IQR", "median", "std"],
        help="Feature names for residual analysis"
    )

    return vars(ap.parse_args())


def save_json(obj, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def main():
    args = parse_arguments()

    dataset_path = args["dataset_path"]
    model_path = args["model_path"]
    out_dir = args["out_dir"]
    period = args["period"]
    eval_split = args["eval_split"]
    feature_names = args["feature_names"]

    makedirs(out_dir, exist_ok=True)

    print("=" * 120)
    print("Analyze trained model")
    print(f"Dataset path : {dataset_path}")
    print(f"Model path   : {model_path}")
    print(f"Output dir   : {out_dir}")
    print(f"Period       : {period}")
    print(f"Eval split   : {eval_split}")
    print("=" * 120)

    # 1. Load dataset
    X_train, y_train, X_test, y_test = read_data_from_dataset(dataset_path)

    if eval_split == "train":
        X_eval = X_train
        y_eval = y_train
    elif eval_split == "test":
        X_eval = X_test
        y_eval = y_test
    else:
        X_eval = np.concatenate([X_train, X_test], axis=0)
        y_eval = np.concatenate([y_train, y_test], axis=0)

    print("X_eval:", X_eval.shape)
    print("y_eval:", y_eval.shape)

    # 2. Build sliding windows
    X_eval_w, y_eval_w = make_sliding_windows(
        X_eval,
        y_eval,
        k=period,
        horizon=1
    )

    print("X_eval_w:", X_eval_w.shape)
    print("y_eval_w:", y_eval_w.shape)

    # 3. Load model
    model = load_model(model_path, custom_objects={"rmse": rmse})

    # 4. Predict
    y_pred = model.predict(X_eval_w, batch_size=1)

    print("y_pred:", y_pred.shape)

    # 5. Save prediction csv
    df_output = pd.DataFrame({
        "True": np.asarray(y_eval_w).reshape(-1),
        "Predicted": np.asarray(y_pred).reshape(-1),
        "Residual": np.asarray(y_eval_w).reshape(-1) - np.asarray(y_pred).reshape(-1)
    })

    output_csv_path = path.join(out_dir, "pred_vs_true.csv")
    df_output.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV saved: {output_csv_path}")

    # 6. Save plots and metrics
    save_prediction_plot(y_eval_w, y_pred, out_dir)
    save_yy_plot(y_eval_w, y_pred, out_dir)

    mse_score, rmse_loss, mae_loss, r2 = save_mse(
        y_eval_w,
        y_pred,
        out_dir,
        model=model
    )

    ResidualPlot(y_eval_w, y_pred, out_dir)
    ErrorHistogram(y_eval_w, y_pred, out_dir)

    # 7. Residual feature analysis
    # 對齊 sliding window 後的最後一個 time step features
    # X_eval_w shape = (samples, period, features)
    # 取每個 window 最後一天的 feature 來分析 residual
    if X_eval_w.shape[-1] == len(feature_names):
        X_last_step = X_eval_w[:, -1, :]

        df_analysis = pd.DataFrame(X_last_step, columns=feature_names)
        df_analysis["true"] = np.asarray(y_eval_w).reshape(-1)
        df_analysis["predicted"] = np.asarray(y_pred).reshape(-1)
        df_analysis["residual"] = df_analysis["true"] - df_analysis["predicted"]
        df_analysis["is_overestimate"] = df_analysis["residual"] < 0

        residual_csv_path = path.join(out_dir, "Residual Analysis.csv")
        df_analysis.to_csv(residual_csv_path, index=False, encoding="utf-8-sig")
        print(f"Residual analysis saved: {residual_csv_path}")

        over = df_analysis[df_analysis["is_overestimate"]]
        under = df_analysis[~df_analysis["is_overestimate"]]

        diff_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean_Overestimate": over[feature_names].mean().values,
            "Mean_Underestimate": under[feature_names].mean().values,
            "Difference": (
                over[feature_names].mean() - under[feature_names].mean()
            ).values
        })

        diff_df["abs_diff"] = diff_df["Difference"].abs()
        diff_df = diff_df.sort_values("abs_diff", ascending=False)

        diff_csv_path = path.join(out_dir, "residual_difference_summary.csv")
        diff_df.to_csv(diff_csv_path, index=False, encoding="utf-8-sig")
        print(f"Residual difference summary saved: {diff_csv_path}")

    else:
        print(
            "Skip residual feature analysis: "
            f"X feature size = {X_eval_w.shape[-1]}, "
            f"feature_names size = {len(feature_names)}"
        )

    # 8. Save summary
    summary = {
        "dataset_path": dataset_path,
        "model_path": model_path,
        "out_dir": out_dir,
        "period": period,
        "eval_split": eval_split,
        "MAE": float(mae_loss),
        "MSE": float(mse_score),
        "RMSE": float(rmse_loss),
        "R2": float(r2)
    }

    save_json(summary, path.join(out_dir, "analysis_summary.json"))

    print("\nAnalysis complete!")
    print(summary)


if __name__ == "__main__":
    main()