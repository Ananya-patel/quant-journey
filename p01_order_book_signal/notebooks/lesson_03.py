import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os

# ── LOAD DATA ─────────────────────────────────────────────────
df = yf.Ticker("AAPL").history(period="1y")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.index = pd.to_datetime(df.index).tz_localize(None)
df['Return'] = df['Close'].pct_change()

# ── ROLLING WINDOWS ───────────────────────────────────────────

# 1. Simple Moving Average (SMA) — average price over last N days
df['SMA_20']  = df['Close'].rolling(window=20).mean()  # 20-day
df['SMA_50']  = df['Close'].rolling(window=50).mean()  # 50-day

# 2. Rolling Volatility — std of returns over last 20 days
# Multiply by sqrt(252) to annualize
df['Vol_20'] = df['Return'].rolling(window=20).std() * np.sqrt(252)

# 3. Rolling Sharpe — the SIGNAL version (20-day rolling)
df['Roll_Sharpe'] = (
    df['Return'].rolling(window=20).mean() * 252 /
    (df['Return'].rolling(window=20).std() * np.sqrt(252))
)

# ── PRINT INSIGHTS ────────────────────────────────────────────
print("Last 10 rows with rolling features:")
print(df[['Close', 'SMA_20', 'SMA_50', 'Vol_20', 'Roll_Sharpe']].tail(10).round(3))

print(f"\nCurrent 20-day volatility (annualized): {df['Vol_20'].iloc[-1]*100:.1f}%")
print(f"Current rolling Sharpe (20-day):        {df['Roll_Sharpe'].iloc[-1]:.2f}")

# When SMA_20 > SMA_50 → uptrend (Golden Cross)
# When SMA_20 < SMA_50 → downtrend (Death Cross)
latest = df.iloc[-1]
if latest['SMA_20'] > latest['SMA_50']:
    print("\nTrend signal: 📈 UPTREND (SMA20 above SMA50)")
else:
    print("\nTrend signal: 📉 DOWNTREND (SMA20 below SMA50)")

# ── VISUALIZE ─────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10))

# Chart 1: Price + Moving Averages
ax1.plot(df.index, df['Close'],  color='white',   lw=1.2, label='Close')
ax1.plot(df.index, df['SMA_20'], color='#00ff9d', lw=1.5, label='SMA 20')
ax1.plot(df.index, df['SMA_50'], color='#ff3c6e', lw=1.5, label='SMA 50')
ax1.set_title('AAPL — Price with Moving Averages')
ax1.legend(); ax1.grid(True, alpha=0.2)

# Chart 2: Rolling Volatility
ax2.plot(df.index, df['Vol_20']*100, color='#ffcc00', lw=1.5)
ax2.axhline(df['Vol_20'].mean()*100, color='white', lw=0.8, linestyle='--', label='Mean vol')
ax2.set_title('Rolling 20-Day Volatility (Annualized %)')
ax2.set_ylabel('%'); ax2.legend(); ax2.grid(True, alpha=0.2)

# Chart 3: Rolling Sharpe
ax3.plot(df.index, df['Roll_Sharpe'], color='#7b61ff', lw=1.5)
ax3.axhline(0, color='white', lw=0.8)
ax3.axhline(1, color='#00ff9d', lw=0.8, linestyle='--', label='Sharpe=1')
ax3.fill_between(df.index, df['Roll_Sharpe'], 0,
                 where=df['Roll_Sharpe'] > 0, alpha=0.2, color='#00ff9d')
ax3.fill_between(df.index, df['Roll_Sharpe'], 0,
                 where=df['Roll_Sharpe'] < 0, alpha=0.2, color='#ff3c6e')
ax3.set_title('Rolling 20-Day Sharpe Ratio')
ax3.legend(); ax3.grid(True, alpha=0.2)

plt.tight_layout()

# Save properly
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, 'lesson_03_rolling.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved!")