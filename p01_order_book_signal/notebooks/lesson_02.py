import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
# ── 1. DOWNLOAD REAL DATA ──────────────────────────────────────
# yfinance pulls from Yahoo Finance — free, real data
# We're getting 1 year of Apple daily prices
ticker = yf.Ticker("AAPL")
df = ticker.history(period="1y")  # 1 year of daily OHLCV

# ── 2. EXPLORE YOUR DATA ──────────────────────────────────────
print("Shape:", df.shape)          # (rows, columns)
print("\nFirst 5 rows:")
print(df.head())                   # always do this first

print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)

# ── 3. CLEAN IT UP ────────────────────────────────────────────
# Keep only what we need for now
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.index = pd.to_datetime(df.index)   # make sure index is datetime
df.index = df.index.tz_localize(None) # remove timezone (cleaner)

print("\nCleaned data — last 5 rows:")
print(df.tail())

# ── 4. CALCULATE DAILY RETURNS ───────────────────────────────
# pct_change() = (today - yesterday) / yesterday
# This is the PANDAS way — works on the whole column at once
df['Return'] = df['Close'].pct_change()

print("\nFirst few returns:")
print(df['Return'].head(10))

# ── 5. KEY STATS ──────────────────────────────────────────────
print("\n── APPLE 1-YEAR STATS ──")
print(f"Trading days:      {len(df)}")
print(f"Avg daily return:  {df['Return'].mean()*100:.3f}%")
print(f"Daily volatility:  {df['Return'].std()*100:.3f}%")
print(f"Best day:          {df['Return'].max()*100:.2f}%")
print(f"Worst day:         {df['Return'].min()*100:.2f}%")

# ── 6. ANNUALIZED STATS (how quants report performance) ───────
import numpy as np
annual_return = df['Return'].mean() * 252      # 252 trading days/year
annual_vol    = df['Return'].std() * np.sqrt(252)
sharpe        = annual_return / annual_vol     # basic Sharpe (no risk-free rate yet)

print(f"\nAnnualized return: {annual_return*100:.2f}%")
print(f"Annualized vol:    {annual_vol*100:.2f}%")
print(f"Sharpe Ratio:      {sharpe:.2f}")

# ── 7. VISUALIZE ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Price chart
ax1.plot(df.index, df['Close'], color='#00ff9d', linewidth=1.5)
ax1.set_title('AAPL — Daily Close Price (1 Year)', fontsize=13)
ax1.set_ylabel('Price ($)')
ax1.grid(True, alpha=0.3)

# Plot 2: Daily returns
ax2.bar(df.index, df['Return']*100,
        color=['#00ff9d' if r > 0 else '#ff3c6e' for r in df['Return']],
        width=1.0, alpha=0.8)
ax2.set_title('AAPL — Daily Returns (%)', fontsize=13)
ax2.set_ylabel('Return (%)')
ax2.axhline(0, color='white', linewidth=0.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)   # creates folder if it doesn't exist
plt.savefig(os.path.join(results_dir, 'lesson_02_aapl.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved to results/")