"""
data_loader.py
──────────────
Fetches Bitcoin (BTC/USDT) hourly data from Binance public API.
No API key needed — this data is completely free.

Why hourly?
  - Daily data: only 365 rows/year — too little for deep learning
  - Hourly data: 8,760 rows/year — enough to train an LSTM
  - Crypto trades 24/7 — no market hours gaps
"""

import requests
import pandas as pd
import numpy as np
import os
import time


def fetch_binance_ohlcv(symbol: str = "BTCUSDT",
                        interval: str = "1h",
                        total_candles: int = 5000) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance public API.

    Args:
        symbol:        trading pair e.g. "BTCUSDT"
        interval:      candle size: "1h", "4h", "1d"
        total_candles: how many candles to fetch (max 1000/request)

    Returns:
        DataFrame with OHLCV columns, datetime index
    """
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    limit    = 1000  # Binance max per request
    fetched  = 0

    print(f"[data_loader] Fetching {total_candles} {interval} "
          f"candles for {symbol}...")

    # We fetch in batches of 1000
    # endTime param lets us paginate backwards
    end_time = None

    while fetched < total_candles:
        params = {
            "symbol":   symbol,
            "interval": interval,
            "limit":    min(limit, total_candles - fetched),
        }
        if end_time:
            params["endTime"] = end_time

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        all_data = batch + all_data
        fetched += len(batch)
        end_time = batch[0][0] - 1  # go further back in time
        time.sleep(0.1)             # be polite to the API

    print(f"[data_loader] Fetched {len(all_data)} candles total")

    # Parse raw API response
    # Binance returns: [openTime, open, high, low, close, volume, ...]
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # Keep only what we need
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    # Rename to match our convention
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Sort chronologically
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)

    print(f"[data_loader] Date range: "
          f"{df.index[0]} → {df.index[-1]}")

    return df


def save_data(df: pd.DataFrame, filename: str) -> None:
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)
    df.to_csv(path)
    print(f"[data_loader] Saved → {path}")


def load_data(filename: str) -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__),
                        '..', 'data', filename)
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    print(f"[data_loader] Loaded {len(df)} rows from CSV")
    return df