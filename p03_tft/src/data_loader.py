"""
data_loader.py
──────────────
Fetches multiple assets for TFT.
TFT is designed for MULTI-ASSET forecasting —
using BTC + ETH together gives richer context.

BTC often leads ETH. If BTC pumps, ETH follows.
The model can learn this cross-asset relationship.
"""

import requests
import pandas as pd
import numpy as np
import time
import os


def fetch_binance_ohlcv(symbol: str,
                        interval: str = "1h",
                        total_candles: int = 5000) -> pd.DataFrame:
    """Fetch OHLCV from Binance. Same as P02."""
    base_url = "https://api.binance.com/api/v3/klines"
    all_data, fetched, end_time = [], 0, None

    print(f"[data_loader] Fetching {symbol}...")

    while fetched < total_candles:
        params = {
            "symbol":   symbol,
            "interval": interval,
            "limit":    min(1000, total_candles - fetched),
        }
        if end_time:
            params["endTime"] = end_time

        r = requests.get(base_url, params=params)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        all_data  = batch + all_data
        fetched  += len(batch)
        end_time  = batch[0][0] - 1
        time.sleep(0.1)

    df = pd.DataFrame(all_data, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','quote_vol','trades',
        'taker_buy_base','taker_buy_quote','ignore'
    ])
    df = df[['timestamp','open','high','low','close','volume']].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col])
    df.columns = ['Open','High','Low','Close','Volume']
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def load_multi_asset(symbols: list,
                     interval: str = "1h",
                     total_candles: int = 6000) -> dict:
    """
    Load multiple assets and align their timestamps.

    Returns dict: {'BTC': df_btc, 'ETH': df_eth, ...}
    All DataFrames share the same index (inner join on time).
    """
    data = {}
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    for symbol in symbols:
        name     = symbol.replace('USDT', '')
        csv_path = os.path.join(data_dir, f'{name}_{interval}.csv')

        if os.path.exists(csv_path):
            print(f"[data_loader] Loading {name} from cache...")
            df = pd.read_csv(csv_path,
                             index_col='timestamp',
                             parse_dates=True)
        else:
            df = fetch_binance_ohlcv(symbol, interval, total_candles)
            df.to_csv(csv_path)
            print(f"[data_loader] Saved {name} → {csv_path}")

        data[name] = df

    # Align all assets to common timestamps
    common_idx = data[list(data.keys())[0]].index
    for name, df in data.items():
        common_idx = common_idx.intersection(df.index)

    for name in data:
        data[name] = data[name].loc[common_idx]

    print(f"\n[data_loader] Aligned {len(symbols)} assets: "
          f"{len(common_idx)} common timestamps")
    print(f"[data_loader] Range: "
          f"{common_idx[0]} → {common_idx[-1]}")

    return data