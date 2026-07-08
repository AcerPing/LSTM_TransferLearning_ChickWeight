from os import path
import pickle

from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager


# matplotlib 字體設定
plt.rcParams["font.size"] = 13
# 設定中文字體，例如 Noto Sans CJK 字體為默認字體
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
chinese_font = font_manager.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = ['Noto Sans CJK SC', chinese_font.get_name()] + matplotlib.rcParams['font.family']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + matplotlib.rcParams['font.sans-serif'] # 其他備選字體


def save_lr_curve(H, out_dir: str, f_name=None):
    """save learning curve in deep learning"""

    f_name = 'Learning Curve' if not f_name else f'{f_name} Learning Curve'

    train_loss = H.history['loss']
    val_loss = H.history['val_loss']
    epochs = range(len(train_loss))

    plt.figure(figsize=(30, 10))
    plt.rcParams["font.size"] = 18

    plt.plot(epochs, train_loss, label='Train', marker='o', markersize=3)
    plt.plot(epochs, val_loss, label='Validation', marker='s', markersize=3)

    # 每張圖最多標約 10 個點，避免文字太密
    label_interval = max(1, len(train_loss) // 10)

    for i, value in enumerate(train_loss):
        if i % label_interval == 0 or i == len(train_loss) - 1:
            plt.annotate(
                f'{value:.4f}',
                xy=(i, value),
                xytext=(0, 6),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=11,
                color='blue',
                alpha=0.8
            )

    for i, value in enumerate(val_loss):
        if i % label_interval == 0 or i == len(val_loss) - 1:
            plt.annotate(
                f'{value:.4f}',
                xy=(i, value),
                xytext=(0, -8),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=11,
                color='orange',
                alpha=0.8
            )

    plt.title(f'{f_name} (Model Loss)', fontsize=18)
    plt.ylabel('MSE Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train', 'Validation'], loc='best', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path.join(out_dir, f'{f_name}.png'), bbox_inches='tight')
    plt.close('all')

    print(f"Plot saved to {path.join(out_dir, f'{f_name}.png')}")


def save_prediction_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str, file_name: str = "prediction.png", title: str = "Comparison of Actual and Predicted Values", y_label: str = "Value"):
    """save prediction plot for tareget varibale

    Args:
        y_test_time (np.array): observed data for target variable # 實際值
        y_pred_test_time (np.array): predicted data for target variable # 預測值
        out_dir (str): directory path for saving # 保存目錄
    """
    plt.figure(figsize=(30, 10)) # 設定圖表大小
    plt.rcParams["font.size"] = 18 # 設置字體大小
    y_true = np.asarray(y_test_time).reshape(-1)
    y_pred = np.asarray(y_pred_test_time).reshape(-1)
    x = np.arange(1, len(y_true) + 1)
    plt.plot(x, y_pred, color='red', label='predicted', marker='x', markersize=2) # 繪製預測數據的紅色折線圖
    plt.plot(x, y_true, color='blue', label='measured', marker="o", markersize=2) # 繪製實際數據的藍色折線圖
    
    # 在每個點上顯示數據標籤 (預測數據)
    for i, value in enumerate(y_pred):
        if i % 1000 == 0:  # 每隔 1000 個數據點顯示一次標籤
            plt.annotate(f'{value:.2f}', xy=(x[i], value), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', color='crimson', fontsize=12, alpha=0.9)
    # 在每個點上顯示數據標籤 (實際數據)
    for i, value in enumerate(y_true):
        if i % 1000 == 0:  # 每隔 1000 個數據點顯示一次標籤
            plt.annotate(f'{value:.2f}', xy=(x[i], value),  xytext=(0, -5), textcoords="offset points", ha='center', va='top', color='dodgerblue', fontsize=12, alpha=0.9)
    
    # -- 動態 y 軸：避免 Stage B / Stage C-1 中 y > 1 或 y_pred < 0 被裁切
    all_values = np.concatenate([
        y_true,
        y_pred
    ])
    y_min = min(0, np.nanmin(all_values))
    y_max = np.nanmax(all_values)
    margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1

    plt.ylim(y_min - margin, y_max + margin) # -- 原本方式 # plt.ylim(0, 1) # 設置y軸的顯示範圍為0到1。
    if len(x) > 1:
        plt.xlim(1, len(x)) # -- 原本方式 # plt.xlim(0, len(y_test_time)) # 設置x軸範圍，從0到實際數據的長度。
    elif len(x) == 1:
        plt.xlim(0.5, 1.5) # -- 原本方式 # plt.xlim(0, len(y_test_time)) # 設置x軸範圍，從0到實際數據的長度。

    plt.title(title) # title = "Comparison of Actual and Predicted Values"
    plt.ylabel(y_label) # plt.ylabel('Value') # 設置y軸標籤。
    plt.xlabel('樣本序列') # 設置x軸標籤。
    plt.legend(loc="best") # 顯示圖例，並將圖例放在最佳位置（由 Matplotlib 自動確定）。
    plt.grid(alpha=0.3)  # 加入透明網格，便於觀察
    plt.tight_layout()
    plt.savefig(path.join(out_dir, file_name), bbox_inches='tight') # 保存圖像
    # plt.show() # 顯示圖表
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(out_dir, file_name)}")


def save_yy_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str, file_name: str = "yy_plot.png", title: str = "Observed vs Predicted Values", axis_label: str = "Value"):
    """save yy plot for target variable

    Args:
        y_test_time (np.array): observed data for target variable # 實際值
        y_pred_test_time (np.array): predicted data for target variable # 預測值
        out_dir (str): directory path for saving # 保存目錄
    """
    plt.figure(figsize=(10, 10)) # 設定圖表大小
    plt.rcParams["font.size"] = 18 # 設置字體大小
    plt.plot(y_test_time, y_pred_test_time, 'b.', label='Predicted vs Observed') # 繪製預測值與實際值的散點圖，使用藍色點標記。
    
    # -- 原本方式
    # diagonal = np.linspace(0, 1, 10000) # 繪製對角線
    # plt.plot(diagonal, diagonal, 'r-', linestyle='--', label='Ideal Line') # 繪製一條紅色的對角線，表示理想狀況下的預測值與實際值相等。散點越接近這條線，表示預測越準確。
    # # Highlight regions for high and low estimates
    # plt.fill_between(diagonal, diagonal, 1, color='peachpuff', alpha=0.2, label='Overestimation Region')  # 淺橙色區域 (color='peachpuff '): 表示模型的高估區域（y_pred_test_time > y_test_time）。
    # plt.fill_between(diagonal, 0, diagonal, color='palegreen', alpha=0.2, label='Underestimation Region')  # 淺綠色區域 (color='peachpuff '): 表示模型的低估區域（y_pred_test_time < y_test_time）。
    # plt.xlim(0, 1) # 設定x軸的範圍為0到1。
    # plt.ylim(0, 1) # 設定y軸的範圍為0到1。
    # -- 動態座標範圍：避免 observed / predicted 超出 0~1 時被裁切
    all_values = np.concatenate([
        np.asarray(y_test_time).reshape(-1),
        np.asarray(y_pred_test_time).reshape(-1)
    ])
    v_min = min(0, np.nanmin(all_values))
    v_max = np.nanmax(all_values)
    margin = (v_max - v_min) * 0.1 if v_max > v_min else 0.1
    axis_min = v_min - margin
    axis_max = v_max + margin
    diagonal = np.linspace(axis_min, axis_max, 1000)
    plt.plot(diagonal, diagonal, 'r--', label='Ideal Line')
    # 高估區：predicted > observed
    plt.fill_between(
        diagonal,
        diagonal,
        axis_max,
        color='peachpuff',
        alpha=0.2,
        label='Overestimation Region'
    )
    # 低估區：predicted < observed
    plt.fill_between(
        diagonal,
        axis_min,
        diagonal,
        color='palegreen',
        alpha=0.2,
        label='Underestimation Region'
    )
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.title(title, fontsize=16) # title = '觀察值與預測值的對角線分析圖'
    plt.xlabel(f'Observed ({axis_label})', fontsize=14) # plt.xlabel('Observed', fontsize=14) # 設置x軸標籤。
    plt.ylabel(f'Predicted ({axis_label})', fontsize=14) # plt.ylabel('Predicted', fontsize=14) # 設置y軸標籤。
    plt.legend(loc='best', fontsize=12) # Add legend
    plt.tight_layout()
    plt.savefig(path.join(out_dir, file_name), bbox_inches='tight') # 保存圖像
    # plt.show()
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(out_dir, file_name)}")


# 自訂RMSE函數 (內部函數)
def _rmse(mse_loss): # 因為Keras並未內建RMSE作為指標，需要自行定義一個自訂的RMSE指標函數。
    '''
    RMSE 是 mse 的平方根，更直觀地表示誤差，與實際數據單位一致。
    '''
    rmse_loss = np.sqrt(mse_loss)
    return rmse_loss


def save_mse(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str, model=None):
    """save mean squared error for tareget variable

    Args:
        y_test_time (np.array): observed data for target variable # 實際值
        y_pred_test_time (np.array): predicted data for target variable # 預測值
        out_dir (str): directory path for saving # 保存目錄
        model : trained model (keras)
    """
    # sklearn
    mse_loss = mse(y_test_time, y_pred_test_time) # 計算均方誤差
    rmse_loss = _rmse(mse_loss)  # RMSE
    mae_loss = mae(y_test_time, y_pred_test_time) # 計算平均絕對誤差
    r2 = r2_score(y_test_time, y_pred_test_time)  # R-squared指標，反映模型解釋目標變數變異程度的能力。

    with open(path.join(out_dir, 'log.txt'), 'w') as f: # 寫入文件
        f.write('MAE預測誤差值 : {:.6f}\n'.format(mae_loss))
        f.write('MSE預測誤差值 : {:.6f}\n'.format(mse_loss))
        f.write('RMSE預測誤差值 : {:.6f}\n'.format(rmse_loss))        
        f.write('R2 Score : {:.6f}\n'.format(r2))
        f.write('=' * 65 + '\n')
        if model:
            model.summary(print_fn=lambda x: f.write(x + '\n')) # 將模型摘要資訊寫入文件。
            
    return mse_loss, rmse_loss, mae_loss, r2


# 殘差圖（Residual Plot）
def ResidualPlot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str, file_name: str = "Residual Plot.png", title: str = "Residual Plot 殘差圖", x_label: str = "Predicted Values", y_label: str = "Residuals"):
    plt.figure(figsize=(12, 8))

    # 統一轉成 1D，避免 broadcasting 錯誤
    y_true = np.asarray(y_test_time).reshape(-1)
    y_pred = np.asarray(y_pred_test_time).reshape(-1)
    residuals = y_true - y_pred # 計算殘差：正值代表模型低估，負值代表模型高估
    plt.scatter(y_pred, residuals, color='blue', alpha=0.6, label='Residuals (y_true - y_pred)') # 繪製殘差散點圖
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Ideal Line (Residual = 0)') # 基準線 (Residual = 0)
    
    # 動態座標範圍
    x_min = min(0, np.nanmin(y_pred))
    x_max = np.nanmax(y_pred)
    x_margin = (x_max - x_min) * 0.1 if x_max > x_min else 0.1

    y_min = np.nanmin(residuals)
    y_max = np.nanmax(residuals)
    y_abs_max = max(abs(y_min), abs(y_max))
    y_margin = y_abs_max * 0.1 if y_abs_max > 0 else 0.1

    axis_x_min = x_min - x_margin
    axis_x_max = x_max + x_margin
    axis_y_min = -y_abs_max - y_margin
    axis_y_max = y_abs_max + y_margin

    x_fill = np.linspace(axis_x_min, axis_x_max, 500)
    # 標示模型低估與高估的區域
    # Residual > 0：模型低估
    plt.fill_between( x_fill, 0, axis_y_max, color='lightgreen', alpha=0.2, label='Underestimation Region (Residual > 0)' ) # 淺綠色 (color='lightgreen'): 模型低估區域（殘差 > 0）。
    # Residual < 0：模型高估
    plt.fill_between( x_fill, axis_y_min, 0, color='lightsalmon', alpha=0.2, label='Overestimation Region (Residual < 0)' )  # 淺橙色 (color='lightsalmon'): 模型高估區域（殘差 < 0）。
    plt.xlim(axis_x_min, axis_x_max)  # 設定X軸，預測值範圍0到1。 # plt.xlim(0, 1)
    plt.ylim(axis_y_min, axis_y_max)  # 設定Y軸，殘差範圍-1到1。 # plt.ylim(-1, 1)
    plt.title(title, fontsize=16) # title = 'Residual Plot 殘差圖'
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=True, edgecolor='black', fancybox=True) # 增加圖例
    plt.tight_layout()
    plt.savefig(path.join(out_dir, file_name), bbox_inches='tight') # 保存圖像
    # plt.show()
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(out_dir, file_name)}")
    

# 誤差直方圖（Error Histogram）
def ErrorHistogram(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str, file_name: str = "Error Histogram.png", title: str = "Error Histogram 誤差直方圖", x_label: str = "Residuals"):
    plt.figure(figsize=(12, 8))
    # 統一轉成 1D，避免 broadcasting 錯誤
    y_true = np.asarray(y_test_time).reshape(-1)
    y_pred = np.asarray(y_pred_test_time).reshape(-1)
    residuals = y_true - y_pred # 計算殘差：正值代表模型低估，負值代表模型高估
    r_min = np.nanmin(residuals)
    r_max = np.nanmax(residuals)
    # # 以 0 為中心建立對稱範圍，讓圖比較穩定好讀；加一點 margin，並確保 0 被包含進顯示範圍
    r_abs_max = max(abs(r_min), abs(r_max))
    margin = r_abs_max * 0.15 if r_abs_max > 0 else 0.1
    axis_min = -r_abs_max - margin
    axis_max = r_abs_max + margin
    # 小樣本時不要用 bins='auto'，改用固定數量
    n = len(residuals)
    bin_count = max(5, min(12, n + 1))
    bins = np.linspace(axis_min, axis_max, bin_count)
    # 填充Overestimation區域（Residual < 0）
    plt.axvspan(axis_min, 0, color='burlywood', alpha=0.2, label='OverEstimation Region (Residual < 0)')  # 當 Residual < 0 時，表示實際值 < 預測值（高估）。    
    # 填充Underestimation區域（Residual > 0）
    plt.axvspan(0, axis_max, color='darkseagreen', alpha=0.2, label='UnderEstimation Region (Residual > 0)')  # 當 Residual > 0 時，表示實際值 > 預測值（低估）。    
    
    plt.hist(residuals, bins=bins, color='deepskyblue', alpha=0.5, edgecolor='black', linewidth=0.8, label='Residuals (y_true - y_pred)') # 柱狀圖，並讓Matplotlib自動計算bins區間。 # -- 'auto'    
    # 每個 residual 加上小點，讓小樣本分布更清楚
    plt.scatter(residuals, np.full_like(residuals, -0.03), color='black', s=35, alpha=0.75, marker='x', label='Individual Residual Points' )
    # Residual = 0 基準線
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Ideal Line (Residual = 0)') # 基準線，加一條紅色的垂直基準線，位於Residual為0的地方。
    plt.xlim(axis_min, axis_max)
    plt.title(title, fontsize=16) # title = 'Error Histogram 誤差直方圖'
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='best', fontsize=12, frameon=True, edgecolor='black', fancybox=True)
    plt.tight_layout()
    plt.savefig(path.join(out_dir, file_name), bbox_inches='tight') # 保存圖像
    # plt.show()
    plt.close('all')  # 關閉所有繪圖對象
    print(f"Plot saved to {path.join(out_dir, file_name)}")


def load_target_scaler(data_dir_path: str):
    """
    讀取 target_scaler.pkl。
    用於將 normalized y_true / y_pred 還原回原始體重尺度。
    """
    scaler_path = path.join(data_dir_path, "target_scaler.pkl")

    if not path.exists(scaler_path):
        print(f"[Original-scale metrics] 找不到 target_scaler.pkl，略過 inverse transform：{scaler_path}")
        return None

    with open(scaler_path, "rb") as f:
        target_scaler = pickle.load(f)

    print(f"[Original-scale metrics] 已讀取 target scaler：{scaler_path}")
    return target_scaler


def save_original_scale_metrics(y_true_norm, y_pred_norm, data_dir_path: str, out_dir: str, args: dict):
    """
    將 normalized y_true / y_pred inverse_transform 回原始體重尺度，
    並輸出 original-scale MAE / MSE / RMSE / R2。
    """
    target_scaler = load_target_scaler(data_dir_path)

    if target_scaler is None:
        return args

    # 保證 shape 為 (N, 1)
    y_true_norm_2d = np.asarray(y_true_norm).reshape(-1, 1)
    y_pred_norm_2d = np.asarray(y_pred_norm).reshape(-1, 1)

    # inverse transform
    y_true_original = target_scaler.inverse_transform(y_true_norm_2d).reshape(-1)
    y_pred_original = target_scaler.inverse_transform(y_pred_norm_2d).reshape(-1)

    # 輸出 original-scale prediction plot，供正式報告 / 簡報使用
    save_prediction_plot(
        y_true_original,
        y_pred_original,
        out_dir,
        file_name="prediction_original_scale.png",
        title="Comparison of Actual and Predicted Values (Original Scale)",
        y_label="Body Weight (g)"
    )
    # 輸出 original-scale yy plot，供正式報告 / 簡報使用
    save_yy_plot(
        y_true_original,
        y_pred_original,
        out_dir,
        file_name="yy_plot_original_scale.png",
        title="Observed vs Predicted Values (Original Scale)",
        axis_label="Body Weight (g)"
    )
    ResidualPlot(
        y_true_original,
        y_pred_original,
        out_dir,
        file_name="residual_plot_original_scale.png",
        title="Residual Plot (Original Scale)",
        x_label="Predicted Body Weight (g)",
        y_label="Residuals (g)"
    )
    ErrorHistogram(
        y_true_original,
        y_pred_original,
        out_dir,
        file_name="error_histogram_original_scale.png",
        title="Error Histogram (Original Scale)",
        x_label="Residuals (g)"
    )

    # 計算 original-scale metrics
    mae_original = mae(y_true_original, y_pred_original)
    mse_original = mse(y_true_original, y_pred_original)
    rmse_original = np.sqrt(mse_original)
    r2_original = r2_score(y_true_original, y_pred_original)

    print("\n[Original-scale metrics]")
    print(f"MAE_original  : {mae_original:.6f}")
    print(f"MSE_original  : {mse_original:.6f}")
    print(f"RMSE_original : {rmse_original:.6f}")
    print(f"R2_original   : {r2_original:.6f}")

    # 存成 CSV，方便後續做表格
    original_pred_path = path.join(out_dir, "prediction_compare.csv")
    np.savetxt(
        original_pred_path,
        np.column_stack([
            y_true_norm_2d.reshape(-1),
            y_pred_norm_2d.reshape(-1),
            y_true_norm_2d.reshape(-1) - y_pred_norm_2d.reshape(-1),
            y_true_original,
            y_pred_original,
            y_true_original - y_pred_original
        ]),
        delimiter=",",
        header="y_true_norm,y_pred_norm,residual_norm,y_true_original,y_pred_original,residual_original",
        comments="",
        fmt="%.6f"
    )

    # 存成 txt
    original_log_path = path.join(out_dir, "metrics_original_scale.txt")
    with open(original_log_path, "w", encoding="utf-8") as f:
        f.write("[Original-scale metrics]\n")
        f.write(f"MAE_original  : {mae_original:.6f}\n")
        f.write(f"MSE_original  : {mse_original:.6f}\n")
        f.write(f"RMSE_original : {rmse_original:.6f}\n")
        f.write(f"R2_original   : {r2_original:.6f}\n")

    # 寫入 params.json
    args["MAE Original"] = float(mae_original)
    args["MSE Original"] = float(mse_original)
    args["RMSE Original"] = float(rmse_original)
    args["R2 Original"] = float(r2_original)

    return args

