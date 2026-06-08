"""
1. 找到指定資料夾
2. 讀取該資料夾中的所有 .pkl 檔
3. 顯示檔名
4. 顯示資料型態
5. 顯示 shape
6. 嘗試把 1 維或 2 維資料轉成 pandas DataFrame
7. 印出完整資料
8. 統計 X_train.pkl + X_test.pkl 的總樣本數與特徵數
"""

import pickle
from pathlib import Path
import pandas as pd

# pandas 完整顯示設定
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

data_dir = project_root / "dataset" / "target" / "FwuSow_Poultry_0_31_fit0_31_full_scaler"

print("正在檢查資料夾：")
print(data_dir)
print("=" * 60)

if not data_dir.exists():
    raise FileNotFoundError(f"找不到資料夾：{data_dir}")

n_features = 0
n_samples = 0

for file_path in data_dir.glob("*.pkl"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"\n檔案名稱：{file_path.name}")
    print(f"資料型態：{type(data)}")

    if hasattr(data, "shape"):
        print(f"資料形狀：{data.shape}")

    print("完整資料：")

    try:
        if hasattr(data, "ndim") and data.ndim == 1:
            df = pd.DataFrame(data, columns=["value"])
            print(df)

        elif hasattr(data, "ndim") and data.ndim == 2:
            df = pd.DataFrame(data)
            print(df)

        else:
            print(data)

    except Exception:
        print(data)

    if file_path.name in ["X_train.pkl", "X_test.pkl"]:
        if hasattr(data, "shape") and len(data.shape) >= 2:
            n_samples += data.shape[0]
            n_features = data.shape[1]

print("\n" + "-" * 60)
print(f"Samples : {n_samples}")
print(f"Features: {n_features}")