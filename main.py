import random
import argparse # 解析命令列參數
import json
# import shutil
import os
from os import path, getcwd, makedirs, environ, listdir
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # ! 不要初始化GPU裝置，避免 CUDA、cuDNN 相容性問題。

import tensorflow as tf
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping 

from utils.model import build_model, rmse
from utils.data_io import (
    read_data_from_dataset,
    # ReccurentTrainingGenerator,
    # ReccurentPredictingGenerator,
    decompose_time_series
)
from utils.save import (
    save_lr_curve,
    save_prediction_plot,
    save_yy_plot,
    save_mse,
    ResidualPlot,
    ErrorHistogram,
    save_original_scale_metrics
)
from utils.device import limit_gpu_memory # 限制 TensorFlow 對 GPU 記憶體的預留或使用量。
from notebook.make_sliding_windows import make_sliding_windows
from reports.Record_args_while_training import Record_args_while_training # 紀錄訓練時的nb_batch、bsize、period


def parse_arguments():
    ap = argparse.ArgumentParser(
        description='Time-Series Regression by LSTM through transfer learning') # 表示該程式的用途是透過遷移學習使用 LSTM 進行時間序列回歸分析。
    
    ap.add_argument('--period', default=3, type=int, help='number of time steps used as input window')
    
    # for dataset path
    ap.add_argument('--out-dir', '-o', default='result', type=str, help='path for output directory') # 指定輸出目錄的路徑，預設值為 result。
    
    # for model
    ap.add_argument('--seed', type=int, default=1234, help='seed value for random value, (default : 1234)') # 確保隨機操作（如資料分割、模型初始化等）在每次執行中一致，方便實驗重現性。
    # ap.add_argument('--train-ratio', default=0.7, type=float, help='percentage of train data to be loaded (default : 0.7)') # 指定訓練集比例為 0.7（即 70%）。數據集會依據此比例分割為訓練集和測試集或驗證集。

    # for training
    ap.add_argument('--train-mode', '-m', default='pre-train', type=str, help='"pre-train", "transfer-learning", "without-transfer-learning" (default : pre-train)') # 設定模式
    ap.add_argument("--pre-model-path", default=None, type=str)
    ap.add_argument("--source-name", default=None, type=str)
    ap.add_argument("--target-name", default=None, type=str)
    ap.add_argument('--gpu', action='store_true', help='whether to do calculations on gpu machines (default : False)') # 是否啟用GPU加速 # ! 因TensorFlow版本套件，暫不啟用GPU。
    ap.add_argument('--nb-epochs', '-e', default=1, type=int, help='training epochs for the model') # 設定訓練的epoch。（epoch是完整地使用所有訓練數據訓練模型的一次過程。）
    ap.add_argument('--nb-batch', default=16, type=int, help='number of batches in training (default : 16)') # 設定訓練過程中的批次數量，預設為 16。 批次大小（batch size） = 總訓練樣本數量 ÷ 批次數量（nb-batch）
    # ap.add_argument('--nb-subset', default=10, type=int,
    #                 help='number of data subset in bootstrapping (default : 10)') # 在bootstrapping中(即Bagging集成式學習)設定資料子集的數量。EX. 生成 10 個不同的訓練子集。
    ap.add_argument('--valid-ratio', default=0.3, type=float, help='ratio of validation data in train data (default : 0.3)') # 在訓練資料中設定驗證資料的比例。
    ap.add_argument('--freeze', action='store_true', help='whether to freeze transferred weights in transfer learning (default : False)') # 在遷移學習中凍結已轉移的權重。
    ap.add_argument('--learning-rate', default=None, type=float, help='initial learning rate for optimizer')
    
    # for output
    ap.add_argument('--train-verbose', default=1, type=int, help='whether to show the learning process (default : 1)') # 設定訓練過程中的輸出詳盡程度。
    args = vars(ap.parse_args())
    return args
    

def seed_every_thing(seed=1234): # 確保各種隨機操作（如資料分割、模型初始化等）在每次執行中產生相同的結果，從而提高實驗的可重現性。
    environ['PYTHONHASHSEED'] = str(seed) # 設定Python的雜湊隨機種子，確保Python的雜湊行為在每次執行時保持一致。
    np.random.seed(seed) # 設定NumPy的隨機種子，確保 NumPy 產生的隨機數在每次執行時相同。
    random.seed(seed) # 設定Python標準庫的隨機種子，確保Python標準庫中的隨機數生成器在每次執行時產生相同的結果。
    tf.random.set_random_seed(seed) # 設定TensorFlow的隨機種子，確保TensorFlow產生的隨機數在每次執行時一致。


def save_arguments(args, out_dir): # 旨在將參數字典 args 以 JSON 格式保存到指定的輸出目錄 out_dir 中
    path_arguments = path.join(out_dir, 'params.json')
    with open(path_arguments, mode="w") as f:
        json.dump(args, f, indent=4)


def make_callbacks(file_path, save_csv=True):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=4, min_lr=1e-7) # 降低學習率，以促進模型更好地收斂。
    model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True, verbose=1) # 保存最佳模型。 # -- save_weights_only = True,
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, min_delta=1e-4, restore_best_weights=True, verbose=1) 
    if not save_csv:
        return [reduce_lr, model_checkpoint, early_stopping]
    csv_logger = CSVLogger(path.join(path.dirname(file_path), 'epoch_log.csv')) # 將每個訓練週期的損失和評估指標記錄到 CSV 文件中
    return [reduce_lr, model_checkpoint, csv_logger, early_stopping] 


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def main():

    # make analysis environment
    limit_gpu_memory() # 限制GPU記憶體使用量，避免因為分配過多而造成系統不穩定。然而當使用量超出設定的限制後，仍然可能發生OOM錯誤。
    args = parse_arguments() # 解析參數
    seed_every_thing(args["seed"]) # 設定隨機種子，在每次運行時產生一致的結果。
    write_out_dir = path.normpath(path.join(getcwd(), 'reports', args["out_dir"])) # 輸出文件的存放路徑
    makedirs(write_out_dir, exist_ok=True)
    
    print('-' * 140)
    print(f'train_mode: {args["train_mode"]} \n')
    print(f'period: {args["period"]}')
    
    if args["train_mode"] == 'pre-train': # 以預訓練模式執行模型訓練。
        
        source_list = listdir("dataset/source")
        if args["source_name"] is not None:
            source_list = [args["source_name"]]
        
        for source in source_list: # for source in listdir('dataset/source'): # 逐個處理來源數據集

            # skip source dataset without pickle file
            data_dir_path = path.join('dataset', 'source', source)
            if not path.exists(f'{data_dir_path}/X_train.pkl'): 
                print(f"Skip source dataset without X_train.pkl: {source}")
                continue
            
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], source)
            makedirs(write_result_out_dir, exist_ok=True)
            
            # load dataset
            X_train, y_train, X_test, y_test = read_data_from_dataset(data_dir_path) # 讀取'X_train', 'y_train', 'X_test', 'y_test'資料
            period = args["period"]  # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。
            # pre-train 階段的目標：不是為了做準確的預測或模型評估，而是為了讓模型學到通用的時序結構、模式或特徵，以便未來可以把學到的權重遷移到另一個任務（也就是 transfer learning 的 target 任務）。
            # 這個資料集（source domain）只是用來初始化權重。模型表現如何，不是我們關心的；而是它「能否幫助另一個資料集」更快收斂、準確預測。
            # pre-train不需要保留 test來做泛化評估，只是學習「時序結構」。

            print("Before merge:")
            print("X_train:", X_train.shape)
            print("y_train:", y_train.shape)
            print("X_test :", X_test.shape)
            print("y_test :", y_test.shape)

            X_train = np.concatenate((X_train, X_test), axis=0)  # > no need for test data when pre-training
            y_train = np.concatenate((y_train, y_test), axis=0)  # > no need for test data when pre-training

            print("After merge:")
            print("X_train:", X_train.shape)
            print("y_train:", y_train.shape)
            
            decomp_result, best_period = decompose_time_series(y_train) # 針對 y_train 做 time series decomposition
            trend = decomp_result["trend"]
            seasonal = decomp_result["period"]
            resid = decomp_result["resid"]

            plt.figure(figsize=(12, 8))
            # 趨勢
            plt.subplot(3, 1, 1)
            plt.plot(trend)
            plt.title("Trend")
            # 季節
            plt.subplot(3, 1, 2)
            plt.plot(seasonal)
            plt.title("Seasonal")
            # 殘差
            plt.subplot(3, 1, 3)
            plt.plot(resid)
            plt.title("Residual")

            plt.tight_layout()
            # plt.show()
            print(f"Decomposition 結果最佳 period: {best_period}") # 這裡的答案會是1
            # period = best_period # 要不要用 best_period？
            # print(f"決定使用 period={period}")
            
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False) # 不隨機打亂數據 (shuffle=False)
            print(f'\nSource dataset : {source}')
            print(f'\nX_train : {X_train.shape[0]}')
            print(f'\nX_valid : {X_valid.shape[0]}')
            print(f'切分比例: {args["valid_ratio"]}')
            print(f'period:{period}') # , args["nb_batch"]: {args["nb_batch"]}
            # print(f'是否X_valid會等於X_test: {X_valid == X_test}')

            # --- 用 sliding windows 展開 ---
            # 訓練階段：用 make_sliding_windows 先把 (X, y) 攤平成 (samples, timesteps=k, features)
            X_train_w, y_train_w = make_sliding_windows(X_train, y_train, k=period, horizon=1)
            X_valid_w, y_valid_w = make_sliding_windows(X_valid, y_valid, k=period, horizon=1)
            print("Train windows:", X_train_w.shape, y_train_w.shape)
            print("Valid windows:", X_valid_w.shape, y_valid_w.shape)

            # construct the model
            file_path = path.join(write_result_out_dir, 'best_model.hdf5') # 指定模型的保存路徑
            callbacks = make_callbacks(file_path) # 在訓練過程中保存最佳模型
            input_shape = (period, X_train.shape[1]) # (timesteps, features)，period表示時間步數，X_train.shape[1]為欄位特徵。
            model = build_model(input_shape, args["gpu"], write_result_out_dir)
            
            # train the model
            bsize = min(16, len(y_train_w)) # 自動計算批次大小 min(16, len(y_train_w)) 會依照資料大小自動調整，確保「每個 epoch 大約有 args["nb_batch"] 個 batch」。
            print(f'nb_batch:{args["nb_batch"]}')
            print(f'批次大小batch_size: {bsize}')
            
            # RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1) # 創建訓練數據
            # RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=1, timesteps=period, delay=1) # 創建驗證數據
            # validation_data = RVG

            print('開始訓練model模型（Pre-Train）')
            Record_args_while_training(write_out_dir, args["train_mode"], source, args['nb_batch'], bsize, period, data_size=(len(y_train) + len(y_valid)))
            # H = model.fit_generator(RTG, validation_data=validation_data, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks) # 訓練模型
            H = model.fit(
                X_train_w, y_train_w, # X_train_w、y_train_w 已經是 numpy array
                validation_data=(X_valid_w, y_valid_w),
                batch_size=bsize,
                epochs=args["nb_epochs"],
                verbose=1,
                callbacks=callbacks
            )
            print(H.history.keys())
            save_lr_curve(H, write_result_out_dir, source) # 保存每個epoch的學習曲線

            
            # --- pre-train 階段驗證集預測 ---

            # --- 方法 1：sliding windows ---
            # X_valid_w, y_valid_w = make_sliding_windows(X_valid, y_valid, k=period, horizon=1) # # 直接用 sliding windows 生成驗證集的輸入 (同訓練一致)
            # 預測
            best_model = load_model(file_path, custom_objects={'rmse': rmse}) # 載入 validation loss 最佳的模型進行 Pre-Train 評估
            y_valid_pred = best_model.predict(X_valid_w, batch_size=1) # # 使用與訓練一致的 sliding windows 做預測；輸入：完整的 numpy array / tensor
            y_valid_eval = y_valid_w 
            print("y_valid_pred:", y_valid_pred.shape)
            print("y_valid_eval :", y_valid_eval.shape)

            # --- 方法 2：ReccurentPredictingGenerator ---
            # 預測階段：仍保留 ReccurentPredictingGenerator，因為它逐筆滑動、能確保預測的時間點和原始序列對齊，方便畫圖對比。
            # RPG = ReccurentPredictingGenerator(X_valid, batch_size=1, timesteps=period) # 生成測試數據。
            #                                                                            # 預測階段，設定 batch_size=1 是為了逐筆預測資料，針對每一筆時間點資料逐一進行預測。
            #                                                                            # 且 單筆預測時，回傳結果可以直接對應到原始 X_valid 中的每個時間點，方便畫圖與對比。
            # y_valid_pred = model.predict_generator(RPG) # 預測測試數據。輸入：一個 Python generator 或 keras.utils.Sequence 類別。
            # y_valid = y_valid[-len(y_valid_pred):] # 將 y_valid 的長度調整為與 y_valid_pred（模型預測值）的長度一致，確保在進行計算和可視化時，兩者長度相符。
            
            # save log for the model (計算誤差指標並保存結果) 保存結果
            save_prediction_plot(y_valid_eval, y_valid_pred, write_result_out_dir) # -- y_valid # 繪製 y_valid 與 y_valid_pred 的對比圖，展示預測值與實際值的偏差 (折線圖)
            save_yy_plot(y_valid_eval, y_valid_pred, write_result_out_dir) # -- y_valid # 繪製 y_valid 與y_valid_pred的對比圖，展示預測值與實際值的偏差 (散點圖)
            mse_score, rmse_loss, mae_loss, r2 = save_mse(y_valid_eval, y_valid_pred, write_result_out_dir, model=best_model) # -- y_valid # 計算 y_valid 和 y_valid_pred 之間的均方誤差（MSE）分數，同時將模型摘要資訊寫入文件。
            # 紀錄參數
            args["MAE Loss"] = mae_loss
            args["MSE Loss"] = mse_score
            args["RMSE Loss"] = rmse_loss
            args["R2 Score"] = r2
            args = save_original_scale_metrics( y_valid_eval, y_valid_pred, data_dir_path, write_result_out_dir, args) # inverse transform 回原始尺度
            Learning_Rate = best_model.optimizer.get_config()["learning_rate"] # 取得最終學習率
            args["Learning Rate"] = Learning_Rate
            save_arguments(args, write_result_out_dir) # 保存訓練參數 (args) 到結果輸出目錄中。
            # 誤差圖
            ResidualPlot(y_valid_eval, y_valid_pred, write_result_out_dir) # -- y_valid
            ErrorHistogram(y_valid_eval, y_valid_pred, write_result_out_dir) # -- y_valid

            # clear memory up (清理記憶體並保存參數)
            keras.backend.clear_session() # 清理記憶體，釋放模型佔用的資源。
            print('\n' * 2 + '-' * 140 + '\n' * 2)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    elif args["train_mode"] == 'transfer-learning': # 使用遷移學習來訓練模型，從預訓練模型中提取權重並應用於新數據集。
        
        target_list = listdir("dataset/target")
        if args["target_name"] is not None:
            target_list = [args["target_name"]]
        
        for target in target_list:
        
            # skip target in the absence of pickle file
            if not path.exists(f'dataset/target/{target}/X_train.pkl'):
                msg = f"找不到 target dataset 或 X_train.pkl：dataset/target/{target}"
                if args["target_name"] is not None:
                    raise FileNotFoundError(msg)
                print(f"Skip target dataset without X_train.pkl: {target}")
                continue

            # for source in listdir(f'{write_out_dir}/pre-train'): # 遍歷預訓練的模型，對每個模型進行遷移學習。
            # 若有指定 --pre-model-path，代表使用外部指定模型，不需要依賴 reports/.../pre-train 資料夾
            if args["pre_model_path"] is not None:
                source_list = [args["source_name"] if args["source_name"] is not None else "custom_pre_model"]
            else:
                source_pretrain_dir = f'{write_out_dir}/pre-train'
                if not path.exists(source_pretrain_dir):
                    raise FileNotFoundError(f"找不到 pre-train 資料夾：{source_pretrain_dir}")

                source_list = listdir(source_pretrain_dir)
                if args["source_name"] is not None:
                    source_list = [args["source_name"]]
                            
            for source in source_list:
                if args["pre_model_path"] is not None:
                    pre_model_path = args["pre_model_path"]
                else:
                    pre_model_path = f'{write_out_dir}/pre-train/{source}/best_model.hdf5' # 確保預訓練模型權重存在。
                if not path.exists(pre_model_path): 
                    print(f"Skip: 找不到 pre_model_path：{pre_model_path}")
                    continue

                # make output directory 保存結果的目錄
                if args["freeze"]:
                    print(f'在遷移學習中，是否凍結權重: {args["freeze"]}，即凍結權重。')
                    train_mode = f'{args["train_mode"]} (Freeze)'
                else:
                    print(f'在遷移學習中，是否凍結權重: {args["freeze"]}，即解凍權重。')
                    train_mode = f'{args["train_mode"]} (Unfreeze)'
                write_result_out_dir = path.join(write_out_dir, train_mode, target, source)
                makedirs(write_result_out_dir, exist_ok=True)
                    
                # load dataset (加載目標數據集)
                data_dir_path = f'dataset/target/{target}'
                period = args["period"] # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。
                X_train, y_train, X_test, y_test = read_data_from_dataset(data_dir_path)
                # 僅從 train 切出 validation
                X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False) # 將訓練集分割為訓練和驗證。
                # 再用 sliding windows 展開
                X_train_w, y_train_w = make_sliding_windows(X_train, y_train, k=period, horizon=1)
                X_valid_w, y_valid_w = make_sliding_windows(X_valid,  y_valid,  k=period, horizon=1)
                print("Train windows:", X_train_w.shape, y_train_w.shape)
                print("Valid windows:", X_valid_w.shape, y_valid_w.shape)

                print(f'\nTarget dataset : {target}')
                print(f'\nSource dataset : {source}')
                print(f'\nX_train : {X_train.shape[0]}')
                print(f'\nX_valid : {X_valid.shape[0]}')
                print(f'\nX_test : {X_test.shape[0]}')
                print(f'period:{period}') # , args["nb_batch"]: {args["nb_batch"]}
                
                # construct the model (構建並編譯模型)
                pre_model = load_model(pre_model_path, custom_objects={'rmse': rmse}) # 加載預訓練模型的權重。
                file_path = path.join(write_result_out_dir, f'{source}_transferred_best_model.hdf5')
                callbacks = make_callbacks(file_path)
                input_shape = (period, X_train.shape[1]) # (timesteps, features)，period表示時間步數，X_train.shape[1]為欄位特徵。
                print('開始建立模型（Transfer-Learning）')
                model = build_model(input_shape, args["gpu"], write_result_out_dir, pre_model=pre_model, freeze=args["freeze"], learning_rate=args["learning_rate"]) # 構建遷移學習模型 # freeze參數決定是否凍結預訓練模型的層，以避免在遷移學習中微調它們。
        
                # train the model (訓練模型)
                bsize = min(16, len(y_train_w)) # 自動計算批次大小 min(16, len(y_train_w)) 會依照資料大小自動調整，確保「每個 epoch 大約有 args["nb_batch"] 個 batch」。
                                                # 調小 Batch Size，提升權重更新的靈敏度，並幫助模型適應新的資料特徵分佈。
                print(f'nb_batch:{args["nb_batch"]}')
                print(f'計算批次大小batch_size: {bsize}')
                # -- RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1) # 生成訓練數據，以批次形式提供給模型。
                # -- RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1) # 生成驗證數據，以批次形式提供給模型。
                Record_args_while_training(write_out_dir, train_mode, target, args['nb_batch'], bsize, period, data_size=(len(y_train) + len(y_valid) + len(y_test)))
                # H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks) # 訓練模型
                H = model.fit(
                    X_train_w, y_train_w,
                    validation_data=(X_valid_w, y_valid_w),
                    epochs=args["nb_epochs"],
                    batch_size= bsize,
                    callbacks=callbacks,
                    verbose=1
                )
                print(H.history.keys())
                save_lr_curve(H, write_result_out_dir, target) # 繪製學習曲線
                
                # prediction (進行預測並保存結果)
                best_model = load_model(file_path, custom_objects={'rmse': rmse}) # 載入 validation loss 最佳的 transferred model
                
                # --- Test set evaluation：正式 TL 評估 ---
                # --- 方法 1：sliding windows ---
                # 與 Without Transfer Learning 使用相同 test set sliding window 評估方式
                X_test_w, y_test_w = make_sliding_windows(X_test, y_test, k=period, horizon=1) # 直接用 sliding windows 生成驗證集的輸入 (同訓練一致)
                y_test_pred = best_model.predict(X_test_w, batch_size=1)
                y_test_eval = y_test_w

                print("X_test_w:", X_test_w.shape)
                print("y_test_pred:", y_test_pred.shape)
                print("y_test_eval :", y_test_eval.shape)

                # --- 方法 2：ReccurentPredictingGenerator ---
                # RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period) # 生成測試數據。
                #                                                                            # 預測階段，設定 batch_size=1 是為了逐筆預測資料，針對每一筆時間點資料逐一進行預測。
                #                                                                            # 且 單筆預測時，回傳結果可以直接對應到原始 X_test 中的每個時間點，方便畫圖與對比。
                # y_test_pred = best_model.predict_generator(RPG) # 預測測試數據
                # y_test = y_test[-len(y_test_pred):] # 將y_test的長度調整為與 y_test_pred（模型預測值）的長度一致，確保在進行計算和可視化時，兩者長度相符。。

                # save log for the model (計算MSE並保存結果)
                save_prediction_plot(y_test_eval, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (折線圖)
                save_yy_plot(y_test_eval, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (散點圖)
                mse_score, rmse_loss, mae_loss, r2 = save_mse(y_test_eval, y_test_pred, write_result_out_dir, model=best_model) # 計算y_test和y_test_pred之間的均方誤差（MSE）分數，同時將模型摘要資訊寫入文件。
                args["MAE Loss"] = mae_loss
                args["MSE Loss"] = mse_score
                args["RMSE Loss"] = rmse_loss
                args["R2 Score"] = r2
                args = save_original_scale_metrics( y_test_eval, y_test_pred, data_dir_path, write_result_out_dir, args) # inverse transform 回原始尺度
                Learning_Rate = best_model.optimizer.get_config()["learning_rate"]
                args["Learning Rate"] = Learning_Rate
                save_arguments(args, write_result_out_dir) # 保存本次訓練或測試的所有參數設定及結果。
                # 誤差圖
                ResidualPlot(y_test_eval, y_test_pred, write_result_out_dir)
                ErrorHistogram(y_test_eval, y_test_pred, write_result_out_dir)

                # clear memory up (清理記憶體並保存參數)
                keras.backend.clear_session() # 釋放記憶體
                print('\n' * 2 + '-' * 140 + '\n' * 2)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    elif args["train_mode"] == 'without-transfer-learning': # 不使用遷移學習

        target_list = listdir("dataset/target")
        if args["target_name"] is not None:
            target_list = [args["target_name"]]
        
        for target in target_list: # for target in listdir('dataset/target'):
        
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], target)
            makedirs(write_result_out_dir, exist_ok=True)

            # load dataset (加載數據集並分割為訓練和驗證集)
            data_dir_path = path.join('dataset', 'target', target)
            if not path.exists(f'{data_dir_path}/X_train.pkl'):
                print(f"Skip target dataset without X_train.pkl: {target}")
                continue

            X_train, y_train, X_test, y_test = read_data_from_dataset(data_dir_path) # 讀取'X_train', 'y_train', 'X_test', 'y_test'資料
            period = args["period"] # period：表示時間步數（time steps），即模型一次看多少步的歷史數據來進行預測。
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False) # 不隨機打亂數據 (shuffle=False)
            print(f'\nTarget dataset : {target}')
            print(f'\nX_train : {X_train.shape[0]}')
            print(f'\nX_valid : {X_valid.shape[0]}')
            print(f'\nX_test : {X_test.shape[0]}')
            print(f'period:{period}') # , args["nb_batch"]: {args["nb_batch"]}

            # --- 用 sliding windows 展開 ---
            X_train_w, y_train_w = make_sliding_windows(X_train, y_train, k=period, horizon=1)
            X_valid_w, y_valid_w = make_sliding_windows(X_valid, y_valid, k=period, horizon=1)
            print("Train windows:", X_train_w.shape, y_train_w.shape) # Train windows: (14, 1, 8) (14,) → 共有 14 個訓練樣本（windows），每個樣本的時間步長（period = 1），每個時間點的特徵數量（features = 8）。
                                                                      # 原始訓練資料：X_train = 15，但是當設定period = 1，代表每個 window 需要 1 筆歷史資料來預測下一筆，所以，15 筆資料只能切出 14 個可用的 (X, y) pair。
            print("Valid windows:", X_valid_w.shape, y_valid_w.shape) # Valid windows: (6, 1, 8) (6,)
            
            # construct the model (構建模型)
            file_path = path.join(write_result_out_dir, 'best_model.hdf5')
            callbacks = make_callbacks(file_path)
            input_shape = (period, X_train.shape[1])
            model = build_model(input_shape, args["gpu"], write_result_out_dir)
            
            # train the model (訓練模型)
            bsize =  min(16, len(y_train_w)) # 自動計算批次大小 min(16, len(y_train_w)) 會依照資料大小自動調整，確保「每個 epoch 大約有 args["nb_batch"] 個 batch」。
            print(f'nb_batch:{args["nb_batch"]}')
            print(f'計算批次大小batch_size: {bsize}')
            # RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1) # 生成訓練數據，以批次形式提供給模型。
            # RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1) # 生成驗證數據，以批次形式提供給模型。
            
            print('開始訓練model模型（Without-Transfer-Learning）')
            Record_args_while_training(write_out_dir, args["train_mode"], target, args['nb_batch'], bsize, period, data_size=(len(y_train) + len(y_valid) + len(y_test)))
            # H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks) # 訓練模型
            H = model.fit(
                X_train_w, y_train_w, # X_train_w、y_train_w 已經是 numpy array
                validation_data=(X_valid_w, y_valid_w),
                batch_size=bsize,
                epochs=args["nb_epochs"],
                verbose=1,
                callbacks=callbacks
            )
            print(H.history.keys())
            save_lr_curve(H, write_result_out_dir, target) # 繪製學習曲線

            # prediction (預測)
            best_model = load_model(file_path, custom_objects={'rmse': rmse}) # 傳遞rmse自定義指標
            
            # --- 方法 1：sliding windows ---
            X_test_w, y_test_w = make_sliding_windows(X_test, y_test, k=period, horizon=1) # 直接用 sliding windows 生成驗證集的輸入 (同訓練一致)
            y_test_pred = best_model.predict(X_test_w, batch_size=1) 
            y_test_eval = y_test_w
            print("X_test_w:", X_test_w.shape)
            print("y_test_pred:", y_test_pred.shape)
            print("y_test_eval :", y_test_eval.shape)
            
            # --- 方法 2：ReccurentPredictingGenerator ---
            # RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period) # 生成測試數據。
            #                                                                            # 預測階段，設定 batch_size=1 是為了逐筆預測資料，針對每一筆時間點資料逐一進行預測。
            #                                                                            # 且 單筆預測時，回傳結果可以直接對應到原始 X_test 中的每個時間點，方便畫圖與對比。            
            # y_test_pred = best_model.predict_generator(RPG) # 預測測試數據
            # y_test = y_test[-len(y_test_pred):] # 將y_test的長度調整為與 y_test_pred（模型預測值）的長度一致，確保在進行計算和可視化時，兩者長度相符。
      
            # save log for the model (計算MSE誤差和保存結果)
            save_prediction_plot(y_test_eval, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (折線圖)
            save_yy_plot(y_test_eval, y_test_pred, write_result_out_dir) # 繪製y_test與y_test_pred的對比圖，展示預測值與實際值的偏差 (散點圖)
            mse_score, rmse_loss, mae_loss, r2 = save_mse(y_test_eval, y_test_pred, write_result_out_dir, model=best_model) # 計算y_test和y_test_pred之間的均方誤差（MSE）分數，
            args["MAE Loss"] = mae_loss
            args["MSE Loss"] = mse_score
            args["RMSE Loss"] = rmse_loss
            args["R2 Score"] = r2
            args = save_original_scale_metrics( y_test_eval, y_test_pred, data_dir_path, write_result_out_dir, args) # inverse transform 回原始尺度
            Learning_Rate = best_model.optimizer.get_config()["learning_rate"] # 取得最終學習率
            args["Learning Rate"] = Learning_Rate
            save_arguments(args, write_result_out_dir) # 保存本次訓練或測試的所有參數設定及結果。
            # 誤差圖
            ResidualPlot(y_test_eval, y_test_pred, write_result_out_dir)
            ErrorHistogram(y_test_eval, y_test_pred, write_result_out_dir)

            # clear memory up (清理記憶體)
            keras.backend.clear_session()
            print('\n' * 2 + '-' * 140 + '\n' * 2)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      
    else:
        print('No matchining train_mode')

if __name__ == '__main__':
    main()
    print('----------- Complete! -----------')
