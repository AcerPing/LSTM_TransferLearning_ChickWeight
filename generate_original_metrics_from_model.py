import argparse
import json
from os import path, makedirs

import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split

from utils.model import rmse
from utils.data_io import read_data_from_dataset
from utils.save import (
    save_original_scale_metrics,
    save_prediction_plot,
    save_yy_plot,
    save_mse,
    ResidualPlot,
    ErrorHistogram
)
from notebook.make_sliding_windows import make_sliding_windows


def parse_arguments():
    ap = argparse.ArgumentParser(
        description="Generate plots and original-scale metrics from an existing trained model."
    )

    ap.add_argument("--dataset-path", required=True, type=str)
    ap.add_argument("--model-path", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)

    ap.add_argument("--period", default=3, type=int)
    ap.add_argument(
        "--eval-mode",
        default="test",
        choices=["test", "pretrain-valid"],
        help="test: use X_test/y_test; pretrain-valid: reproduce pre-train validation split"
    )
    ap.add_argument("--valid-ratio", default=0.3, type=float)

    return vars(ap.parse_args())


def main():
    args = parse_arguments()

    dataset_path = args["dataset_path"]
    model_path = args["model_path"]
    out_dir = args["out_dir"]
    period = args["period"]

    makedirs(out_dir, exist_ok=True)

    if not path.exists(dataset_path):
        raise FileNotFoundError(f"找不到 dataset path: {dataset_path}")

    if not path.exists(model_path):
        raise FileNotFoundError(f"找不到 model path: {model_path}")

    print("=" * 100)
    print("Generate plots and original-scale metrics from existing model")
    print(f"Dataset path : {dataset_path}")
    print(f"Model path   : {model_path}")
    print(f"Output dir   : {out_dir}")
    print(f"Period       : {period}")
    print(f"Eval mode    : {args['eval_mode']}")
    print("=" * 100)

    X_train, y_train, X_test, y_test = read_data_from_dataset(dataset_path)

    if args["eval_mode"] == "pretrain-valid":
        # 對應 main.py 的 pre-train 評估方式：
        # X_train + X_test 合併後，再切 validation
        X_all = np.concatenate((X_train, X_test), axis=0)
        y_all = np.concatenate((y_train, y_test), axis=0)

        X_train_split, X_eval, y_train_split, y_eval = train_test_split(
            X_all,
            y_all,
            test_size=args["valid_ratio"],
            shuffle=False
        )

    else:
        # 對應 without-transfer-learning / transfer-learning：
        # 使用 test set 評估
        X_eval = X_test
        y_eval = y_test

    print("X_eval:", X_eval.shape)
    print("y_eval:", y_eval.shape)

    X_eval_w, y_eval_w = make_sliding_windows(
        X_eval,
        y_eval,
        k=period,
        horizon=1
    )

    if len(X_eval_w) == 0:
        raise ValueError(
            f"sliding window 數量為 0，請檢查 period={period} 是否大於 evaluation data 長度。"
        )

    print("X_eval_w:", X_eval_w.shape)
    print("y_eval_w:", y_eval_w.shape)

    best_model = load_model(model_path, custom_objects={"rmse": rmse})
    # # 4. Predict
    y_pred = best_model.predict(X_eval_w, batch_size=1)

    # 4-1. Save normalized-scale prediction CSV
    # normalized_csv_path = path.join(
    #     out_dir,
    #         f"prediction_normalized_scale_{args['eval_mode']}.csv"
    #     )
    # np.savetxt(
    #     normalized_csv_path,
    #     np.column_stack([
    #         np.asarray(y_eval_w).reshape(-1),
    #         np.asarray(y_pred).reshape(-1),
    #         np.asarray(y_eval_w).reshape(-1) - np.asarray(y_pred).reshape(-1)
    #     ]),
    #     delimiter=",",
    #     header="y_true_norm,y_pred_norm,residual_norm",
    #     comments="",
    #     fmt="%.6f"
    # )

    # 5. Save normalized-scale plots and metrics / 重新輸出 normalized-scale 圖表與指標
    
    save_prediction_plot(
        y_eval_w,
        y_pred,
        out_dir,
        file_name="prediction_normalized_scale.png",
        title="Comparison of Actual and Predicted Values (Normalized Scale)",
        y_label="Normalized Value"
    ) # save_prediction_plot(y_eval_w, y_pred, out_dir)
    save_yy_plot(
        y_eval_w,
        y_pred,
        out_dir,
        file_name="yy_plot_normalized_scale.png",
        title="Observed vs Predicted Values (Normalized Scale)",
        axis_label="Normalized Value"
    )   # save_yy_plot(y_eval_w, y_pred, out_dir)
    save_mse(y_eval_w, y_pred, out_dir, model=best_model)
    ResidualPlot(
        y_eval_w,
        y_pred,
        out_dir,
        file_name="residual_plot_normalized_scale.png",
        title="Residual Plot (Normalized Scale)",
        x_label="Predicted Values (Normalized)",
        y_label="Residuals (Normalized)"
    ) # ResidualPlot(y_eval_w, y_pred, out_dir)
    ErrorHistogram(
        y_eval_w,
        y_pred,
        out_dir,
        file_name="error_histogram_normalized_scale.png",
        title="Error Histogram (Normalized Scale)",
        x_label="Residuals (Normalized)"
    ) # ErrorHistogram(y_eval_w, y_pred, out_dir)
    print("y_pred:", y_pred.shape)
    # 6. Save original-scale metrics
    metric_args = {
        "dataset_path": dataset_path,
        "model_path": model_path,
        "period": period,
        "eval_mode": args["eval_mode"]
    }

    metric_args = save_original_scale_metrics(
        y_eval_w,
        y_pred,
        dataset_path,
        out_dir,
        metric_args
    )

    with open(path.join(out_dir, "original_metrics_params.json"), "w", encoding="utf-8") as f:
        json.dump(metric_args, f, indent=4, ensure_ascii=False)

    print("\nDone.")
    metrics_path = path.join(out_dir, "metrics_original_scale.txt")
    compare_path = path.join(out_dir, "prediction_compare.csv")
    if path.exists(metrics_path):
        print(f"Saved: {metrics_path}")
    else:
        print(f"Skipped: {metrics_path}")

    if path.exists(compare_path):
        print(f"Saved: {compare_path}")
    else:
        print(f"Skipped: {compare_path}")


if __name__ == "__main__":
    main()