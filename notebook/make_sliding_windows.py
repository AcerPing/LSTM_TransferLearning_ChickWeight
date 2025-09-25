import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def make_sliding_windows(X, y, k=5, horizon=1, stride=1, drop_last_partial=True, return_index=False):
    """
    X: (T, F) 特徵矩陣（按時間排序）
    y: (T,)   目標序列（同樣按時間）
    k:       視窗長度（timesteps）
    horizon: 預測步長；=1 表示用 t-k..t-1 預測 t
    stride:  視窗滑動步長；=1 為完全重疊
    drop_last_partial: 最尾端不足 k+horizon 的是否丟棄
    return_index: 是否回傳每個樣本的 (start_idx, end_idx)
    回傳:
      Xw: (N, k, F), yw: (N,)，可直接餵入 LSTM
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    T = len(y)
    last_valid = T - (k + horizon) + 1
    if last_valid <= 0:
        raise ValueError(f"序列太短，至少需要 k+horizon={k+horizon} 個時間點，實際 T={T}")

    starts = np.arange(0, last_valid, stride)
    Xw = []
    yw = []
    idx_pairs = []

    for s in starts:
        e = s + k
        t = e + (horizon - 1)          # 目標時間點
        if t >= T:
            if drop_last_partial: break
            else:  # 可選：用 pad/補值處理，這裡先略
                break
        Xw.append(X[s:e, :])
        yw.append(y[t])
        if return_index:
            idx_pairs.append((s, e-1, t))

    Xw = np.stack(Xw, axis=0)
    yw = np.array(yw)
    return (Xw, yw, np.array(idx_pairs)) if return_index else (Xw, yw)
