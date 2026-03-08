from src.data_loader import load_multi_asset
from src.features    import build_all_features

data = load_multi_asset(['BTCUSDT', 'ETHUSDT'],
                        interval='1h',
                        total_candles=6000)

df, feature_cols, target_col = build_all_features(data)

print(f"\nFeatures ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  {f}")
print(f"\nShape: {df.shape}")