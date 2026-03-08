"""
features.py
───────────
Richer feature set for TFT.
Key addition: CROSS-ASSET features.
BTC's behavior predicts ETH — we encode this explicitly.
"""

import pandas as pd
import numpy as np
import ta


def build_asset_features(df: pd.DataFrame,
                         name: str) -> pd.DataFrame:
    """
    Build features for one asset.
    All features are prefixed with asset name.
    e.g. 'BTC_rsi', 'ETH_volume_ratio'
    """
    out = pd.DataFrame(index=df.index)

    # ── RETURNS ───────────────────────────────────────────────
    out[f'{name}_ret_1h']  = df['Close'].pct_change(1)
    out[f'{name}_ret_4h']  = df['Close'].pct_change(4)
    out[f'{name}_ret_24h'] = df['Close'].pct_change(24)
    out[f'{name}_log_ret'] = np.log(df['Close']/df['Close'].shift(1))

    # ── VOLUME ────────────────────────────────────────────────
    out[f'{name}_vol_ratio'] = (df['Volume'] /
                                df['Volume'].rolling(20).mean())

    # ── TECHNICALS ────────────────────────────────────────────
    out[f'{name}_rsi'] = ta.momentum.RSIIndicator(
                            df['Close'], 14).rsi() / 100  # normalize 0-1

    macd = ta.trend.MACD(df['Close'])
    out[f'{name}_macd_diff'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df['Close'], 20)
    out[f'{name}_bb_pos'] = ((df['Close'] - bb.bollinger_lband()) /
                             (bb.bollinger_hband() -
                              bb.bollinger_lband() + 1e-9))

    # ── OBI PROXY ─────────────────────────────────────────────
    hl = (df['High'] - df['Low']).replace(0, np.nan)
    out[f'{name}_obi'] = ((df['Close'] - df['Low']) / hl - 0.5) * 2

    # ── VOLATILITY ────────────────────────────────────────────
    out[f'{name}_vol_24h'] = out[f'{name}_log_ret'].rolling(24).std()

    return out


def build_cross_features(data: dict) -> pd.DataFrame:
    """
    Build CROSS-ASSET features.
    These capture relationships BETWEEN assets.
    """
    cross = pd.DataFrame(index=list(data.values())[0].index)

    if 'BTC' in data and 'ETH' in data:
        btc = data['BTC']['Close']
        eth = data['ETH']['Close']

        # ETH/BTC ratio — when this rises, ETH outperforms
        cross['eth_btc_ratio'] = eth / btc

        # Rolling correlation — when high, assets move together
        btc_ret = btc.pct_change()
        eth_ret = eth.pct_change()
        cross['btc_eth_corr_24h'] = btc_ret.rolling(24).corr(eth_ret)

        # BTC return 1h ago predicts ETH now?
        cross['btc_ret_lag1'] = btc_ret.shift(1)

    return cross


def build_all_features(data: dict,
                       target_asset: str = 'BTC') -> tuple:
    """
    Build full feature matrix and target.

    Returns:
        df_features: all features combined
        feature_cols: list of feature column names
        target_col: name of target column
    """
    all_features = []

    # Per-asset features
    for name, df in data.items():
        feat = build_asset_features(df, name)
        all_features.append(feat)

    # Cross-asset features
    cross = build_cross_features(data)
    all_features.append(cross)

    # Combine
    df_feat = pd.concat(all_features, axis=1)

    # ── TARGET: future 4h return of target asset ──────────────
    target_close   = data[target_asset]['Close']
    future_ret     = target_close.pct_change(4).shift(-4)
    df_feat['target'] = future_ret

    # Drop NaN rows
    df_feat.dropna(inplace=True)

    feature_cols = [c for c in df_feat.columns if c != 'target']

    print(f"[features] {len(feature_cols)} features, "
          f"{len(df_feat)} rows")
    print(f"[features] Target — mean: "
          f"{df_feat['target'].mean()*100:.3f}%, "
          f"std: {df_feat['target'].std()*100:.3f}%")

    return df_feat, feature_cols, 'target'