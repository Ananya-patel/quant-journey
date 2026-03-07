from src.data_loader import fetch_binance_ohlcv, save_data
from src.features    import build_features

# Fetch data
df_raw = fetch_binance_ohlcv("BTCUSDT", "1h", total_candles=3000)
save_data(df_raw, "btc_1h.csv")

# Build features
df, feature_cols = build_features(df_raw)

print("\nFeature preview:")
print(df[feature_cols].tail(5).round(4))
print(f"\nShape: {df.shape}")
print(f"Features: {feature_cols}")