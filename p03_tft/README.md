# P03 — Temporal Fusion Transformer for Crypto Forecasting

**Target firms:** Two Sigma · Citadel · D.E. Shaw · Goldman  
**Difficulty:** Intermediate-Advanced  
**Stack:** Python · PyTorch · pandas · ta

---

## Architecture
```
Input (60h × 23 features) [BTC + ETH + cross-asset]
    ↓
Variable Selection Network   learns feature importance
    ↓
2-Layer LSTM Encoder         temporal encoding
    ↓
Multi-Head Attention (4)     multi-scale patterns
    ↓
Gated Residual Networks      adaptive skip connections
    ↓
Quantile Heads × 3           predicts q10, q50, q90
```

## Key Results

| Metric | Value |
|---|---|
| Direction Accuracy | **53.5%** on test set |
| Quantile Coverage | 68.5% (target: 80%) |
| Pinball Loss | 0.001844 |
| Model Parameters | 415,592 |

## Key Innovations vs P02 LSTM

1. **Multi-asset** — BTC + ETH features + cross-asset signals
2. **Quantile regression** — predicts uncertainty ranges, not just direction
3. **Feature importance** — VSN shows which features the model uses
4. **Interpretable attention** — which timesteps drive predictions

## What I Learned

- Quantile calibration: 68.5% coverage vs 80% target
  → fix with temperature scaling on logits
- Direction accuracy (53.5%) is meaningful signal
  but converting it to profit requires asymmetric
  risk filtering on the quantile outputs
- Cross-asset features (ETH/BTC ratio) ranked highly
  in VSN weights — validating multi-asset approach

## How to Run
```bash
pip install -r requirements.txt
python main.py       # train (~5 min on CPU)
python evaluate.py   # quantile evaluation + backtest
```
```

---

## ✅ P03 Complete — What You've Built So Far
```
P01  ✅  OBI Signal + Backtester
         → market microstructure, vectorized backtesting

P02  ✅  LSTM + Attention (Crypto)
         → deep learning, sequence modeling, 52% accuracy

P03  ✅  Temporal Fusion Transformer
         → quantile regression, feature importance,
           53.5% direction accuracy, 415k parameters

Progression: rule-based → deep learning → transformer
Each project uses the previous one's concepts.
```

---

## 🚀 P04 Preview — Graph Neural Networks
```
NEW MATHEMATICAL CONCEPT: Graphs

Everything so far treated stocks as INDEPENDENT.
P04 treats markets as a NETWORK.

Tesla → affects → Battery stocks
                       ↓
                  Lithium miners
                       ↓
                  Chilean Peso

When correlation structure SHIFTS (like March 2020)
→ the graph changes
→ a GNN detects this BEFORE the price moves

You'll learn:
  - Graph theory basics (nodes, edges, adjacency matrix)
  - Graph Convolution (how information flows on graphs)
  - Graph Attention Networks (GAT)
  - Hidden Markov Models for regime labeling
  - Dynamic graphs (edges change every week)