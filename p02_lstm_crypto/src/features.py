"""
features.py
───────────
Builds input features for the LSTM from raw OHLCV data.

FEATURE CATEGORIES:
  1. Price-based   → returns, log returns, price ratios
  2. Volume-based  → volume change, buy/sell pressure
  3. Technical     → RSI, MACD, Bollinger Bands
  4. Microstructure→ OBI proxy (from P01!), volatility

WHY THESE FEATURES?
  Raw prices are non-stationary (they trend up over time).
  Neural networks learn better from stationary inputs.
  Returns and ratios are stationary.
"""

import pandas as pd
import numpy as np
import ta   # technical analysis library


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from OHLCV data.
    Returns DataFrame with features + target column.
    """
    df = df.copy()

    # ── 1. RETURN FEATURES ──────────────────────────────────
    df['return_1h']  = df['Close'].pct_change(1)
    df['return_4h']  = df['Close'].pct_change(4)
    df['return_24h'] = df['Close'].pct_change(24)

    # Log returns (more stationary than simple returns)
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # ── 2. VOLUME FEATURES ──────────────────────────────────
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma20']   = df['Volume'].rolling(20).mean()
    df['volume_ratio']  = df['Volume'] / df['volume_ma20']

    # ── 3. TECHNICAL INDICATORS ─────────────────────────────
    # RSI — momentum oscillator, ranges 0-100
    # > 70 = overbought, < 30 = oversold
    df['rsi'] = ta.momentum.RSIIndicator(
                    df['Close'], window=14).rsi()

    # MACD — trend following
    macd_ind       = ta.trend.MACD(df['Close'])
    df['macd']     = macd_ind.macd()
    df['macd_sig'] = macd_ind.macd_signal()
    df['macd_diff']= macd_ind.macd_diff()   # histogram

    # Bollinger Bands — volatility bands around price
    bb = ta.volatility.BollingerBands(df['Close'], window=20)
    df['bb_upper']  = bb.bollinger_hband()
    df['bb_lower']  = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    # Where is price within the bands? (0=lower, 1=upper)
    df['bb_position'] = ((df['Close'] - df['bb_lower']) /
                         (df['bb_upper'] - df['bb_lower']))

    # ── 4. OBI PROXY (from P01!) ────────────────────────────
    hl_range      = (df['High'] - df['Low']).replace(0, np.nan)
    df['obi']     = ((df['Close'] - df['Low']) / hl_range - 0.5) * 2
    df['obi_20']  = df['obi'].rolling(20).mean()

    # ── 5. VOLATILITY ───────────────────────────────────────
    df['volatility_24h'] = df['log_return'].rolling(24).std()
    df['volatility_7d']  = df['log_return'].rolling(168).std()

    # ── 6. TARGET VARIABLE ──────────────────────────────────
    # What are we predicting?
    # 1 = price will be higher in 4 hours
    # 0 = price will be lower or flat in 4 hours
    future_return = df['Close'].pct_change(4).shift(-4)
    df['target']  = -1   # -1 = ignore (neutral/flat)
    df.loc[future_return >  0.003, 'target'] = 1   # strong UP
    df.loc[future_return < -0.003, 'target'] = 0   # strong DOWN

# Remove neutral rows — only train on clear signals
    df = df[df['target'] != -1].copy()
    df['target'] = df['target'].astype(int)
    # ── CLEAN UP ────────────────────────────────────────────
    # Drop rows where any feature is NaN
    # (happens at start due to rolling windows)
    df.dropna(inplace=True)

    feature_cols = [
        'return_1h', 'return_4h', 'return_24h', 'log_return',
        'volume_change', 'volume_ratio',
        'rsi', 'macd', 'macd_sig', 'macd_diff',
        'bb_position',
        'obi', 'obi_20',
        'volatility_24h', 'volatility_7d'
    ]

    print(f"[features] Built {len(feature_cols)} features, "
          f"{len(df)} rows remain after cleaning")
    print(f"[features] Target balance: "
          f"{df['target'].mean()*100:.1f}% UP days")

    return df, feature_cols