# P01 — Order Book Imbalance Signal & Backtester

**Target firms:** HRT · Citadel Securities · Virtu  
**Difficulty:** Foundation  
**Stack:** Python · Pandas · NumPy · yfinance · Matplotlib

---

## What This Project Does

Implements a quantitative trading signal based on **Order Book 
Imbalance (OBI)** — a core concept in market microstructure used 
by high-frequency trading firms.

Since real Level-2 order book data requires expensive exchange 
feeds, this project uses a **closing-price-location proxy** that 
captures the same intuition from daily OHLCV data:
```
OBI = ((Close - Low) / (High - Low) - 0.5) × 2
```

A value near +1 means price closed near the high → buying pressure.  
A value near -1 means price closed near the low → selling pressure.

---

## Signal Architecture
```
Raw OHLCV Data
      ↓
OBI Proxy (volume-weighted closing location)
      ↓
Rolling Smoothing (20-day window, reduces noise)
      ↓
Trend Filter (only trade in SMA-50 direction)
      ↓
Volatility Filter (go flat when annualized vol > 40%)
      ↓
Final Signal (+1 / -1 / 0) with 1-day execution lag
      ↓
Backtest Engine (with transaction costs @ 5bps/trade)
```

---

## Results (AAPL, 2-year backtest)

| Metric | OBI Strategy | Buy & Hold |
|---|---|---|
| Sharpe Ratio | see output | ~0.40 |
| Max Drawdown | ~-20% | ~-30% |
| Total Trades | low | 1 |

**Key finding:** The vol filter successfully reduced max drawdown 
from -45% (v1) to -20% (v2). The OBI proxy signal has limited 
predictive power on daily data — real OBI requires tick-level 
order book data. This project establishes the research 
infrastructure used in all subsequent projects.

---

## Project Structure
```
p01_order_book_signal/
├── src/
│   ├── data_loader.py   # data fetching & cleaning
│   ├── signal.py        # OBI computation & filtering
│   └── backtester.py    # vectorized backtest engine
├── data/                # cached CSV data
├── results/             # output charts
├── notebooks/           # learning notebooks (F01-F04)
├── main.py              # entry point
└── README.md
```

---

## How to Run
```bash
git clone 
cd p01_order_book_signal
pip install -r requirements.txt
python main.py
```

---

## Key Concepts Learned

- **Market microstructure** — how prices form from order flow
- **Lookahead bias** — the #1 backtest mistake, and how to prevent it
- **Signal filtering** — trend + volatility regime filters
- **Risk metrics** — Sharpe, Max Drawdown, Calmar Ratio
- **Vectorized backtesting** — production-quality research code

---

## What's Next

P02 uses an **LSTM neural network** on crypto data where 
tick-frequency patterns are stronger and data is free.