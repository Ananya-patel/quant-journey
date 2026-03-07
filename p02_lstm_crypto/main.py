"""
main.py — P02 Entry Point
─────────────────────────
Runs the full pipeline:
  data → features → dataset → model → train → evaluate
"""

from src.data_loader import fetch_binance_ohlcv, save_data, load_data
from src.features    import build_features
from src.dataset     import prepare_dataloaders
from src.model       import CryptoLSTM, count_parameters
from src.trainer     import train_model, plot_training
import os, torch

# ── CONFIG ────────────────────────────────────────────────────
SYMBOL      = "BTCUSDT"
INTERVAL    = "1h"
N_CANDLES   = 8000
SEQ_LEN     = 60       # 60 hours of context
BATCH_SIZE  = 64
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.35
N_EPOCHS    = 40
LR          = 1e-3
PATIENCE    = 10

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  P02 — LSTM + Attention Crypto Predictor")
    print("═"*55)

    # ── 1. DATA ───────────────────────────────────────────────
    csv_path = os.path.join("data", "btc_1h.csv")
    if os.path.exists(csv_path):
        print("[main] Loading cached data...")
        df_raw = load_data("btc_1h.csv")
    else:
        df_raw = fetch_binance_ohlcv(SYMBOL, INTERVAL, N_CANDLES)
        save_data(df_raw, "btc_1h.csv")

    # ── 2. FEATURES ───────────────────────────────────────────
    df, feature_cols = build_features(df_raw)

    # ── 3. DATASET ────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler, info = \
        prepare_dataloaders(df, feature_cols,
                            seq_len=SEQ_LEN,
                            batch_size=BATCH_SIZE)

    # ── 4. MODEL ──────────────────────────────────────────────
    model = CryptoLSTM(
        input_size  = len(feature_cols),
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT
    )

    print(f"\n[main] Model parameters: "
          f"{count_parameters(model):,}")
    print(f"[main] Architecture:\n{model}")

    # ── 5. TRAIN ──────────────────────────────────────────────
    history = train_model(
        model, train_loader, val_loader,
        n_epochs=N_EPOCHS, lr=LR, patience=PATIENCE
    )

    # ── 6. PLOT ───────────────────────────────────────────────
    plot_training(history)

    print("\n[main] P02 training complete!")
    print("[main] Next step: evaluate on test set → say 'next'")
