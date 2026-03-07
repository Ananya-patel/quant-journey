"""
dataset.py
──────────
Prepares data for PyTorch training.

KEY CONCEPTS:
  - Normalization: scale features to similar ranges
    (RSI is 0-100, returns are 0.001 — LSTM struggles with this)
  - Sequence creation: convert flat table into 3D sequences
  - Train/Val/Test split: time-based, never shuffle!
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class CryptoSequenceDataset(Dataset):
    """
    PyTorch Dataset that serves (sequence, label) pairs.

    For each index i, returns:
      X = features from hour (i) to hour (i + seq_len - 1)
      y = target at hour (i + seq_len)

    Shape of X: (seq_len, n_features)  e.g. (60, 15)
    Shape of y: scalar 0 or 1
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(sequences)   # float32
        self.y = torch.LongTensor(targets)      # int64 for classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(features: np.ndarray,
                     targets: np.ndarray,
                     seq_len: int = 60):
    """
    Convert flat feature array into overlapping sequences.

    Example with seq_len=3:
      Input:  [f1, f2, f3, f4, f5, f6]
      Output: [(f1,f2,f3)→t4, (f2,f3,f4)→t5, (f3,f4,f5)→t6]

    This is called a SLIDING WINDOW.
    """
    X, y = [], []

    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])   # seq_len rows
        y.append(targets[i + seq_len])         # label at end

    return np.array(X), np.array(y)


def prepare_dataloaders(df: pd.DataFrame,
                        feature_cols: list,
                        seq_len:    int = 60,
                        batch_size: int = 64,
                        train_pct:  float = 0.70,
                        val_pct:    float = 0.15):
    print(f"[dataset] Input rows after target filtering: {len(df)}")
    print(f"[dataset] Target balance: {df['target'].mean()*100:.1f}% UP")
    """
    Full pipeline: normalize → split → sequence → DataLoader.

    Returns:
        train_loader, val_loader, test_loader, scaler, split_info
    """

    # ── STEP 1: EXTRACT ARRAYS ──────────────────────────────
    X_raw = df[feature_cols].values    # shape: (n_rows, n_features)
    y_raw = df['target'].values        # shape: (n_rows,)

    print(f"[dataset] Raw shapes — X: {X_raw.shape}, y: {y_raw.shape}")

    # ── STEP 2: TIME-BASED SPLIT (before scaling!) ──────────
    # CRITICAL: split BEFORE normalizing
    # If you normalize on all data first, test data leaks into train
    n = len(X_raw)
    train_end = int(n * train_pct)
    val_end   = int(n * (train_pct + val_pct))

    X_train_raw = X_raw[:train_end]
    X_val_raw   = X_raw[train_end:val_end]
    X_test_raw  = X_raw[val_end:]

    y_train = y_raw[:train_end]
    y_val   = y_raw[train_end:val_end]
    y_test  = y_raw[val_end:]

    # ── STEP 3: NORMALIZE ───────────────────────────────────
    # RobustScaler uses median + IQR — handles outliers well
    # (price data has many outliers — better than StandardScaler)
    # FIT only on train data, TRANSFORM all sets
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)   # fit + transform
    X_val   = scaler.transform(X_val_raw)          # transform only
    X_test  = scaler.transform(X_test_raw)         # transform only

    # Save scaler for later use (inference)
    scaler_path = os.path.join(os.path.dirname(__file__),
                               '..', 'models', 'scaler.pkl')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # ── STEP 4: CREATE SEQUENCES ────────────────────────────
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   seq_len)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  seq_len)

    print(f"[dataset] Sequence shapes:")
    print(f"  Train: {X_train_seq.shape} → {y_train_seq.shape}")
    print(f"  Val:   {X_val_seq.shape}   → {y_val_seq.shape}")
    print(f"  Test:  {X_test_seq.shape}  → {y_test_seq.shape}")

    # ── STEP 5: PYTORCH DATASETS & DATALOADERS ──────────────
    train_ds = CryptoSequenceDataset(X_train_seq, y_train_seq)
    val_ds   = CryptoSequenceDataset(X_val_seq,   y_val_seq)
    test_ds  = CryptoSequenceDataset(X_test_seq,  y_test_seq)

    # DataLoader batches the data and shuffles train set
    # shuffle=False for val/test — order matters for time series
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, drop_last=False)

    split_info = {
        'n_train': len(train_ds),
        'n_val':   len(val_ds),
        'n_test':  len(test_ds),
        'seq_len': seq_len,
        'n_features': len(feature_cols),
        'train_target_pct': y_train_seq.mean(),
        'test_target_pct':  y_test_seq.mean(),
    }

    print(f"\n[dataset] DataLoaders ready:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val   batches: {len(val_loader)}")
    print(f"  Test  batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, scaler, split_info