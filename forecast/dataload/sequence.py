import pandas as pd
import numpy as np
def create_sequences_train(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    return np.array(X), np.array(y)
def create_sequences_pre(data, seq_length, pred_length):
    X, y = [], []
    for i in range(0,len(data) - seq_length - pred_length + 1,pred_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    return np.array(X), np.array(y)