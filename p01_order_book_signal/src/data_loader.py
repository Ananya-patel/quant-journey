"""
data_loader.py
──────────────
Responsible for ONE thing only: fetching and cleaning price data.
Single Responsibility Principle — each file does one job.
"""

import pandas as pd 
import yfinance as yf
import os 

def load_price_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download OHLCV data for a given ticker.
    
    Args:
        ticker: Stock symbol e.g. "AAPL"
        period: How far back e.g. "1y", "2y", "5y"
    
    Returns:
        Clean DataFrame with OHLCV + Return columns
    """

    print(f"[data_loader] Downloading {ticker} — {period} of data...")
    raw = yf.Ticker(ticker).history(period=period)
    # Keep only what we need
    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    # Clean index
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = 'Date'
    # Drop any rows with missing values
    df.dropna(inplace=True)
    
    # Daily return
    df['Return'] = df['Close'].pct_change()

     # First row will have NaN return — drop it
    df.dropna(inplace=True)
    
    print(f"[data_loader] Loaded {len(df)} trading days "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    
    return df
def save_data(df: pd.DataFrame, ticker: str) -> None:
    """Save data to CSV so we don't re-download every run."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f'{ticker}_daily.csv')
    df.to_csv(path)
    print(f"[data_loader] Saved to {path}")


def load_from_csv(ticker: str) -> pd.DataFrame:
    """Load previously saved data (faster than re-downloading)."""
    path = os.path.join(os.path.dirname(__file__), '..', 'data',
                        f'{ticker}_daily.csv')
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    print(f"[data_loader] Loaded from CSV: {len(df)} rows")
    return df