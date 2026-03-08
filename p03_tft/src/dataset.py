"""
dataset.py
──────────
Quantile-aware dataset for TFT.
Target is now a CONTINUOUS return value, not binary.
The model will predict the distribution of returns.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import pickle, os


class TFTDataset(Dataset):
    """
    Returns (sequence, target) pairs.
    Target is the actual future return (float).
    """
    def __init__(self, sequences: np.ndarray,
                 targets: np.ndarray):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(features, targets, seq_len=60):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(targets[i+seq_len])
    return np.array(X), np.array(y)


def prepare_dataloaders(df, feature_cols,
                        target_col='target',
                        seq_len=60,
                        batch_size=64,
                        train_pct=0.70,
                        val_pct=0.15):

    X_raw = df[feature_cols].values
    y_raw = df[target_col].values

    n         = len(X_raw)
    train_end = int(n * train_pct)
    val_end   = int(n * (train_pct + val_pct))

    # Split BEFORE scaling
    X_train_r = X_raw[:train_end]
    X_val_r   = X_raw[train_end:val_end]
    X_test_r  = X_raw[val_end:]
    y_train   = y_raw[:train_end]
    y_val     = y_raw[train_end:val_end]
    y_test    = y_raw[val_end:]

    # Scale features only (not target — we want raw returns)
    scaler  = RobustScaler()
    X_train = scaler.fit_transform(X_train_r)
    X_val   = scaler.transform(X_val_r)
    X_test  = scaler.transform(X_test_r)

    # Save scaler
    model_dir = os.path.join(os.path.dirname(__file__),
                             '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Create sequences
    Xtr, ytr = create_sequences(X_train, y_train, seq_len)
    Xva, yva = create_sequences(X_val,   y_val,   seq_len)
    Xte, yte = create_sequences(X_test,  y_test,  seq_len)

    print(f"[dataset] Train: {Xtr.shape} | "
          f"Val: {Xva.shape} | Test: {Xte.shape}")

    train_loader = DataLoader(TFTDataset(Xtr, ytr),
                              batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TFTDataset(Xva, yva),
                              batch_size=batch_size,
                              shuffle=False)
    test_loader  = DataLoader(TFTDataset(Xte, yte),
                              batch_size=batch_size,
                              shuffle=False)

    return train_loader, val_loader, test_loader, scaler, {
        'n_features': len(feature_cols),
        'seq_len':    seq_len,
        'n_train':    len(Xtr),
        'n_test':     len(Xte),
    }