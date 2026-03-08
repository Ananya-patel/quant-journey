"""
main.py — P03 TFT Entry Point
"""
from src.data_loader import load_multi_asset
from src.features    import build_all_features
from src.dataset     import prepare_dataloaders
from src.model       import TemporalFusionTransformer, count_parameters
from src.trainer     import train_model, plot_training

# ── CONFIG ────────────────────────────────────────────────────
SYMBOLS      = ['BTCUSDT', 'ETHUSDT']
INTERVAL     = '1h'
N_CANDLES    = 6000
TARGET_ASSET = 'BTC'
SEQ_LEN      = 60
BATCH_SIZE   = 64
HIDDEN_SIZE  = 64
LSTM_LAYERS  = 2
N_HEADS      = 4
DROPOUT      = 0.1
QUANTILES    = [0.1, 0.5, 0.9]
N_EPOCHS     = 40
LR           = 1e-3
PATIENCE     = 8

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  P03 — Temporal Fusion Transformer")
    print("═"*55)

    # 1. Data
    data = load_multi_asset(SYMBOLS, INTERVAL, N_CANDLES)

    # 2. Features
    df, feature_cols, target_col = build_all_features(
                                        data, TARGET_ASSET)

    # 3. Dataset
    train_loader, val_loader, test_loader, scaler, info = \
        prepare_dataloaders(df, feature_cols, target_col,
                            SEQ_LEN, BATCH_SIZE)

    # 4. Model
    model = TemporalFusionTransformer(
        n_features  = info['n_features'],
        hidden_size = HIDDEN_SIZE,
        lstm_layers = LSTM_LAYERS,
        n_heads     = N_HEADS,
        dropout     = DROPOUT,
        quantiles   = QUANTILES
    )

    print(f"\n[main] Parameters: {count_parameters(model):,}")

    # 5. Train
    history = train_model(
        model, train_loader, val_loader,
        quantiles=QUANTILES,
        n_epochs=N_EPOCHS,
        lr=LR,
        patience=PATIENCE
    )

    # 6. Plot
    plot_training(history)

    print("\n[main] Training complete!")
    print("[main] Say 'next' → evaluate quantiles + "
          "feature importance")