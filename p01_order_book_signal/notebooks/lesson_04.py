import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ══════════════════════════════════════════════════════════════
# PART 1: Simulate a simple order book
# In real life this comes from exchange data feeds
# We simulate to understand the mechanics
# ══════════════════════════════════════════════════════════════

np.random.seed(42)   # for reproducibility — always set this

n_steps = 500        # 500 time steps (think: 500 seconds of trading)

# Simulate bid and ask volumes at best level (random walk style)
# In real markets these fluctuate constantly
bid_volumes = np.abs(np.random.normal(5000, 1500, n_steps)).astype(int)
ask_volumes = np.abs(np.random.normal(5000, 1500, n_steps)).astype(int)

# ── CALCULATE OBI ─────────────────────────────────────────────
obi = (bid_volumes - ask_volumes) / (bid_volumes + ask_volumes)

# ── SIMULATE PRICE THAT RESPONDS TO OBI ──────────────────────
# Price moves up when OBI is positive, down when negative
# This is a simplified version of real market dynamics
price = 260.0
prices = [price]

for i in range(1, n_steps):
    # Price change is driven by OBI + random noise
    noise = np.random.normal(0, 0.05)      # random market noise
    signal = obi[i-1] * 0.10              # OBI drives price
    price = price + signal + noise
    prices.append(price)

prices = np.array(prices)

# ── BUILD A DATAFRAME ────────────────────────────────────────
df = pd.DataFrame({
    'time':       range(n_steps),
    'bid_vol':    bid_volumes,
    'ask_vol':    ask_volumes,
    'obi':        obi,
    'price':      prices,
})

# ── ROLLING OBI (smoother signal) ────────────────────────────
df['obi_smooth'] = df['obi'].rolling(window=10).mean()

# ── SIMPLE SIGNAL: trade based on OBI ────────────────────────
# If OBI > 0.1 → predict price going UP → BUY signal = +1
# If OBI < -0.1 → predict price going DOWN → SELL signal = -1
# Otherwise → no position = 0
df['signal'] = 0
df.loc[df['obi'] >  0.1, 'signal'] = 1    # long
df.loc[df['obi'] < -0.1, 'signal'] = -1   # short

# ── PRINT STATS ───────────────────────────────────────────────
print("Order Book Imbalance — Sample Statistics")
print(f"Mean OBI:     {df['obi'].mean():.4f}")
print(f"Std OBI:      {df['obi'].std():.4f}")
print(f"Max OBI:      {df['obi'].max():.4f}  (strongest buying pressure)")
print(f"Min OBI:      {df['obi'].min():.4f}  (strongest selling pressure)")
print(f"\nBUY  signals: {(df['signal'] == 1).sum()}")
print(f"SELL signals: {(df['signal'] == -1).sum()}")
print(f"FLAT (no trade): {(df['signal'] == 0).sum()}")

print("\nFirst 10 rows:")
print(df[['time','bid_vol','ask_vol','obi','signal']].head(10).round(3))

# ── VISUALIZE ─────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10))
fig.patch.set_facecolor('#0d0d14')
for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#0d0d14')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    ax.yaxis.label.set_color('white')

# Chart 1: Simulated price
ax1.plot(df['time'], df['price'], color='white', lw=1.2)
ax1.set_title('Simulated Price (driven by OBI + noise)')
ax1.set_ylabel('Price ($)'); ax1.grid(True, alpha=0.15)

# Chart 2: OBI
ax2.plot(df['time'], df['obi'], color='#6b6b88', lw=0.8, alpha=0.6, label='Raw OBI')
ax2.plot(df['time'], df['obi_smooth'], color='#00ff9d', lw=1.5, label='Smoothed OBI')
ax2.axhline( 0.1, color='#00ff9d', lw=0.8, linestyle='--')
ax2.axhline(-0.1, color='#ff3c6e', lw=0.8, linestyle='--')
ax2.axhline(0,    color='white',   lw=0.5)
ax2.set_title('Order Book Imbalance (OBI)')
ax2.set_ylabel('OBI'); ax2.legend(); ax2.grid(True, alpha=0.15)

# Chart 3: Signal
colors = ['#00ff9d' if s == 1 else '#ff3c6e' if s == -1 else '#333' for s in df['signal']]
ax3.bar(df['time'], df['signal'], color=colors, width=1.0)
ax3.set_title('Trading Signal (+1=Buy, -1=Sell, 0=Flat)')
ax3.set_ylabel('Signal'); ax3.grid(True, alpha=0.15)

plt.tight_layout()
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, 'lesson_04_obi.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved!")