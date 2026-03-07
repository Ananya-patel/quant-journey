"""
signal.py — v2
──────────────
Upgraded OBI signal with:
1. Longer smoothing (less noise)
2. Trend filter (only trade in direction of trend)
3. Volatility regime filter (avoid high-vol periods)
"""

import pandas as pd
import numpy as np


def compute_obi_proxy(df: pd.DataFrame,
                      smooth_window: int = 5) -> pd.DataFrame:
    df = df.copy()

    hl_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['obi_raw'] = (df['Close'] - df['Low']) / hl_range
    df['obi'] = (df['obi_raw'] - 0.5) * 2

    vol_norm = df['Volume'] / df['Volume'].rolling(20).mean()
    df['obi_weighted'] = df['obi'] * vol_norm

    df['obi_smooth'] = df['obi_weighted'].rolling(
                            window=smooth_window,
                            min_periods=1).mean()
    return df


def compute_signal(df: pd.DataFrame,
                   threshold: float = 0.1,
                   trend_window: int = 50,
                   vol_window:   int = 20,
                   vol_cap:      float = 0.40) -> pd.DataFrame:
    """
    Upgraded signal with 3 filters:

    Filter 1 — TREND FILTER
      Only go LONG when price > SMA50 (uptrend)
      Only go SHORT when price < SMA50 (downtrend)
      Reason: OBI works better when aligned with trend

    Filter 2 — VOLATILITY FILTER
      If 20-day annualized vol > vol_cap → go FLAT
      Reason: High volatility = noisy signal + big losses

    Filter 3 — TRANSACTION COST
      Each trade costs 0.05% (realistic for retail)
    """
    df = df.copy()

    # Trend filter
    df['sma_trend'] = df['Close'].rolling(trend_window).mean()
    df['in_uptrend'] = df['Close'] > df['sma_trend']

    # Volatility filter
    df['vol_20'] = df['Return'].rolling(vol_window).std() * np.sqrt(252)
    df['low_vol'] = df['vol_20'] < vol_cap

    # Base signal from OBI
    df['signal_raw'] = 0
    df.loc[df['obi_smooth'] >  threshold, 'signal_raw'] = 1
    df.loc[df['obi_smooth'] < -threshold, 'signal_raw'] = -1

    # Apply trend filter
    # Long only allowed in uptrend, short only in downtrend
    df['signal_filtered'] = df['signal_raw'].copy()
    df.loc[(df['signal_raw'] ==  1) & (~df['in_uptrend']),  'signal_filtered'] = 0
    df.loc[(df['signal_raw'] == -1) & ( df['in_uptrend']),  'signal_filtered'] = 0

    # Apply volatility filter — go flat when vol too high
    df.loc[~df['low_vol'], 'signal_filtered'] = 0

    # Final signal: shift by 1 (no lookahead bias)
    df['signal'] = df['signal_filtered'].shift(1)
    df.dropna(inplace=True)

    return df