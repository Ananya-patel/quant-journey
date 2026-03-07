# P02 — LSTM + Attention for Crypto Direction Prediction

**Target firms:** Two Sigma · Citadel · D.E. Shaw  
**Difficulty:** Intermediate  
**Stack:** Python · PyTorch · pandas · ta · scikit-learn

---

## Architecture
```
Input (60h × 15 features)
    ↓
2-Layer LSTM (hidden=128)     learns temporal patterns
    ↓
Self-Attention Layer          learns which hours matter most
    ↓
Dropout (0.35)                prevents overfitting
    ↓
Linear → 2                    P(DOWN), P(UP)
```

## Results (BTC/USDT, 8000 hourly candles)

| Metric | Value |
|---|---|
| Test Accuracy | 52.1% |
| ROC-AUC | ~0.52 |
| Strategy MDD | -6.5% |
| BTC Buy&Hold MDD | -15%+ same period |

## Key Finding

Attention weights concentrated on hours 58-60 ago,
suggesting the model learned a ~2.5 day mean reversion 
pattern. Prediction bias toward UP class identified —
next iteration would use weighted cross-entropy loss.

## How to Run
```bash
pip install -r requirements.txt
python main.py       # train
python evaluate.py   # test set evaluation
```
```

---

# 🚀 WHAT'S IN P03 — Temporal Fusion Transformer

Before we start, understand WHY P03 is harder and better:
```
P02 (LSTM):                    P03 (TFT):
────────────────────           ────────────────────
Single asset (BTC)             Multiple assets
All features equal             Learns WHICH features matter
One timeframe                  Multiple timeframes
Binary prediction              Quantile prediction
                               (predicts a RANGE, not just up/down)

TFT = Temporal Fusion Transformer
Google DeepMind paper, 2020
Used by hedge funds for multi-asset forecasting
```

**The key new concept in P03:**
```
Instead of predicting "UP or DOWN" (binary)...

TFT predicts:
  10th percentile: "worst case, price will be here"
  50th percentile: "most likely, price will be here"  
  90th percentile: "best case, price will be here"

This is called QUANTILE REGRESSION.
A trading desk uses this to size positions:
  Wide range → uncertain → small position
  Narrow range → confident → large position