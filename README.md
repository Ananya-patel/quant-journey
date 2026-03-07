# Quant Journey — ML for Quantitative Finance

Building toward a position at a top quant firm
(Jane Street · HRT · Citadel · Two Sigma · D.E. Shaw)

---

## Projects

| # | Project | Concepts | Status |
|---|---------|----------|--------|
| P01 | OBI Signal & Backtester | Market microstructure, Sharpe, Drawdown | ✅ Complete |
| P02 | LSTM + Attention (Crypto) | Deep learning, Attention, Backtesting | ✅ Complete |
| P03 | Temporal Fusion Transformer | Quantile regression, Multi-asset | 🔨 Building |
| P04 | Graph Neural Network | GNN, Regime detection, HMM | ⏳ Upcoming |
| P05 | Deep RL Market Making | PPO, Avellaneda-Stoikov | ⏳ Upcoming |
| P06 | NLP Alpha Signal (SEC) | FinBERT, Factor analysis | ⏳ Upcoming |
| P07 | Neural SDE (Vol Surface) | Stochastic calculus, Options | ⏳ Upcoming |
| P08 | Live Trading System | Full MLOps, Kafka, AWS | ⏳ Upcoming |

---

## Stack
Python · PyTorch · pandas · scikit-learn · 
matplotlib · yfinance · ta · cvxpy

---

## Structure
Each project is self-contained with its own
README, requirements.txt, and results/ folder.
```

Then push again:
```
git add README.md
git commit -m "Add root README"
git push
```

---

## What Your GitHub Will Look Like
```
quant-journey/
├── README.md                
├── p01_order_book_signal/
│   ├── README.md            ← explains P01
│   ├── results/             ← your backtest charts
│   ├── src/                 ← clean code
│   └── main.py
└── p02_lstm_crypto/
    ├── README.md            ← explains P02
    ├── results/             ← training curves + eval charts
    ├── src/                 ← clean code
    └── main.py