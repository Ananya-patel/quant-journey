# P05 — Deep Reinforcement Learning Market Making (PPO)

## Overview
Implements a Proximal Policy Optimization (PPO) agent that learns
to make markets in a simulated limit order book. The agent quotes
bid and ask prices continuously, earning the spread while managing
inventory risk.

## What is Market Making?
A market maker simultaneously quotes buy (bid) and sell (ask) prices.
Profit comes from the spread between them. The risk is inventory
accumulation — holding too many units when price moves against you.
```
Agent quotes:  Bid = Mid - bid_offset
               Ask = Mid + ask_offset

Profit per round trip = bid_offset + ask_offset (the spread)
Risk = inventory × price_move (directional exposure)
```

## Architecture
```
State (8 features)
  [inventory, price_return, spread, volatility,
   pnl_norm, time_remaining, bid_fill_rate, ask_fill_rate]
          ↓
    Actor Network          Critic Network
   (8 → 64 → 64 → 2)      (8 → 64 → 64 → 1)
   outputs Gaussian         outputs V(s)
   distribution             scalar value
          ↓                      ↓
   bid_offset, ask_offset    advantage estimate
          ↓
   PPO Clipped Update
   ratio = π_new / π_old
   clipped to [0.8, 1.2]
```

## Environment
| Parameter       | Value        | Meaning                        |
|-----------------|--------------|--------------------------------|
| Price process   | GBM + jumps  | Geometric Brownian Motion      |
| Volatility      | 1%/day       | Daily annualized ~16%          |
| Order arrivals  | Poisson(15)  | 15 orders/minute average       |
| Fill prob       | exp(-1.5×δ)  | Tighter spread = more fills    |
| Episode length  | 390 steps    | 1 trading day (1 step/minute)  |
| Inventory limit | ±50 units    | Hard position limit            |

## PPO Hyperparameters
| Parameter    | Value | Meaning                              |
|--------------|-------|--------------------------------------|
| γ (discount) | 0.99  | Care about future rewards            |
| λ (GAE)      | 0.95  | Advantage estimation smoothing       |
| ε (clip)     | 0.20  | Max 20% policy change per update     |
| Target KL    | 0.05  | Early stop if policy changes too much|
| LR           | 3e-4  | Adam learning rate                   |
| Episodes     | 1000  | Training episodes                    |

## Results

### PPO Agent vs Random Baseline
| Metric                | PPO Agent | Random |
|-----------------------|-----------|--------|
| Avg PnL/episode       | +$5-7     | ~$0    |
| Spread captured       | ~$5.50    | ~$2.00 |
| Final \|inventory\|   | ~3.5      | ~5.0   |
| PnL consistency       | Positive every episode | Variable |

### What the Agent Learned
✅ Consistently captures bid-ask spread (+$5-7/episode)  
✅ Beats random baseline on spread income (~2.5× more)  
✅ Maintains stable quoting behavior  
❌ Did not learn inventory management (held ~3.5 units)  
❌ Policy converged to fixed spread (no state adaptation)  

## Honest Assessment

### What Worked
- Full PPO from scratch with Actor-Critic architecture
- Gaussian policy for continuous action space
- GAE advantage estimation
- KL early stopping (fixed KL explosion from 4.6 → 0.001)
- Agent earns positive PnL every single episode

### What Didn't Work
- Inventory management — agent holds 3-4 units throughout
- Policy entropy froze at 0.43 — one fixed action learned
- No state-dependent quoting (ignores inventory signal)

### Root Cause
Inventory management requires credit assignment over 389 steps.
With γ=0.99: discount after 389 steps = 0.99^389 ≈ 0.019.
The reward for flattening inventory at episode end is worth
almost nothing at step 1. The agent rationally ignores it.

### What Real Firms Do
- 10M+ environment steps (we used 390,000)
- Episode length 10-20 steps (not 390)
- Shaped rewards giving immediate inventory feedback  
- Curriculum learning: start with no inventory risk, add gradually
- 64+ parallel environments for faster learning



## Files
```
p05_rl_market_making/
├── src/
│   ├── environment.py   # Simulated LOB (GBM + Poisson orders)
│   ├── model.py         # Actor-Critic networks (Gaussian policy)
│   ├── ppo.py           # PPO algorithm with KL early stopping
│   └── trainer.py       # Training loop
├── main.py              # Entry point
├── evaluate.py          # Evaluation + visualization
├── debug.py             # Diagnostic tool
├── results/             # Charts saved here
└── models/              # Trained weights saved here
```

## Run
```bash
pip install -r requirements.txt
python main.py        # Train agent (1000 episodes, ~5 min)
python evaluate.py    # Evaluate + generate charts
```
```

---

---

