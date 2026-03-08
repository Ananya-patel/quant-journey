"""
evaluate.py — P03 TFT
──────────────────────
Evaluates the TFT on test set with:
1. Quantile prediction plots (the key TFT output)
2. Feature importance from VSN weights
3. Attention heatmap (which timesteps matter)
4. Confidence-based trading backtest
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, pickle

from src.data_loader import load_multi_asset
from src.features    import build_all_features
from src.dataset     import prepare_dataloaders
from src.model       import TemporalFusionTransformer

# ── CONFIG (must match main.py) ───────────────────────────────
SEQ_LEN     = 60
BATCH_SIZE  = 64
HIDDEN_SIZE = 64
LSTM_LAYERS = 2
N_HEADS     = 4
DROPOUT     = 0.1
QUANTILES   = [0.1, 0.5, 0.9]

# ── LOAD DATA ─────────────────────────────────────────────────
print("\n" + "═"*55)
print("  P03 — TFT Evaluation")
print("═"*55)

data = load_multi_asset(['BTCUSDT','ETHUSDT'], '1h', 6000)
df, feature_cols, target_col = build_all_features(data, 'BTC')

_, _, test_loader, scaler, info = prepare_dataloaders(
    df, feature_cols, target_col, SEQ_LEN, BATCH_SIZE)

# ── LOAD MODEL ────────────────────────────────────────────────
model = TemporalFusionTransformer(
    n_features  = info['n_features'],
    hidden_size = HIDDEN_SIZE,
    lstm_layers = LSTM_LAYERS,
    n_heads     = N_HEADS,
    dropout     = DROPOUT,
    quantiles   = QUANTILES
)
model.load_state_dict(
    torch.load('models/best_tft.pt', map_location='cpu'))
model.eval()
print("[eval] Model loaded ✅")

# ── INFERENCE ─────────────────────────────────────────────────
all_q10, all_q50, all_q90 = [], [], []
all_targets  = []
all_vsn      = []   # variable selection weights
all_attn     = []   # attention weights

with torch.no_grad():
    for X, y in test_loader:
        preds, vsn_w, attn_w = model(X)

        all_q10.extend(preds[:, 0].numpy())
        all_q50.extend(preds[:, 1].numpy())
        all_q90.extend(preds[:, 2].numpy())
        all_targets.extend(y.numpy())
        all_vsn.extend(vsn_w.numpy())
        all_attn.extend(attn_w.numpy())

all_q10     = np.array(all_q10)
all_q50     = np.array(all_q50)
all_q90     = np.array(all_q90)
all_targets = np.array(all_targets)
all_vsn     = np.array(all_vsn)    # (n, seq_len, n_features)
all_attn    = np.array(all_attn)   # (n, seq_len, seq_len)

# ── METRICS ───────────────────────────────────────────────────
# Quantile coverage: what % of actuals fall within q10-q90?
# A well-calibrated model should have ~80% coverage
in_interval = ((all_targets >= all_q10) &
               (all_targets <= all_q90))
coverage    = in_interval.mean()

# Direction accuracy using q50 median prediction
direction_acc = (np.sign(all_q50) ==
                 np.sign(all_targets)).mean()

# Interval width (narrower = more confident)
interval_width = (all_q90 - all_q10).mean()

# MAE on median prediction
mae = np.abs(all_q50 - all_targets).mean()

print(f"\n{'═'*45}")
print(f"  TEST SET METRICS")
print(f"{'═'*45}")
print(f"  Quantile Coverage (q10-q90): {coverage*100:.1f}%")
print(f"  Target coverage:              ~80%")
print(f"  Direction Accuracy (q50):    {direction_acc*100:.1f}%")
print(f"  Median MAE:                  {mae*100:.4f}%")
print(f"  Avg Interval Width:          {interval_width*100:.4f}%")
print(f"{'═'*45}")

# Coverage interpretation
if coverage >= 0.75:
    print(f"\n  ✅ Model is WELL-CALIBRATED")
    print(f"     Uncertainty estimates are reliable")
elif coverage >= 0.65:
    print(f"\n  ⚠️  Model is SLIGHTLY UNDER-CONFIDENT")
else:
    print(f"\n  ❌ Model is OVER-CONFIDENT")
    print(f"     Prediction intervals are too narrow")

# ── CONFIDENCE-BASED BACKTEST ─────────────────────────────────
# KEY INSIGHT: trade ONLY when model is confident
# Confidence = NARROW prediction interval (q90 - q10)
# Narrow interval → model sees clear pattern → bigger edge

n_test   = len(all_targets)
test_idx = df.index[-(n_test):]
test_btc = data['BTC']['Close'].reindex(test_idx)

results = pd.DataFrame({
    'actual':    all_targets,
    'q10':       all_q10,
    'q50':       all_q50,
    'q90':       all_q90,
    'width':     all_q90 - all_q10,
}, index=test_idx[:n_test])

# Signal logic:
# Only trade when interval is NARROW (confident prediction)
# Direction comes from q50 (median forecast)
# ── CONFIDENCE-BASED BACKTEST ─────────────────────────────────
n_test   = len(all_targets)
test_idx = df.index[-n_test:]

results = pd.DataFrame({
    'actual': all_targets,
    'q10':    all_q10,
    'q50':    all_q50,
    'q90':    all_q90,
    'width':  all_q90 - all_q10,
}, index=test_idx)

# ── SIGNAL (asymmetric risk filtering) ───────────────────────
MIN_EXPECTED = 0.0001    # q50 > 0.01%
MAX_DOWNSIDE = -0.0050   # q10 > -0.5%
MIN_UPSIDE   =  0.0005   # q90 > 0.05%
RISK_REWARD  =  1.2      # easier to satisfy

# BEST SIGNAL: use q50 for direction + width for confidence
# Only trade when BOTH conditions met:
# 1. q50 has conviction (direction signal)
# 2. Width is NARROW (model is confident)

width_median = results['width'].median()
THRESHOLD    = 0.0002

results['signal'] = 0

confident = results['width'] < width_median  # narrow = confident

results.loc[confident & (results['q50'] >  THRESHOLD), 'signal'] =  1
results.loc[confident & (results['q50'] < -THRESHOLD), 'signal'] = -1

print(f"\nSignal counts:")
print(f"  Long:  {(results['signal'] ==  1).sum()}")
print(f"  Short: {(results['signal'] == -1).sum()}")
print(f"  Flat:  {(results['signal'] ==  0).sum()}")
# Add this TEMPORARILY after results DataFrame is created
# to understand our data range
print("\nData ranges:")
print(f"  q50 range:   {results['q50'].min()*100:.3f}% to {results['q50'].max()*100:.3f}%")
print(f"  q10 range:   {results['q10'].min()*100:.3f}% to {results['q10'].max()*100:.3f}%")
print(f"  q90 range:   {results['q90'].min()*100:.3f}% to {results['q90'].max()*100:.3f}%")
print(f"  width range: {results['width'].min()*100:.3f}% to {results['width'].max()*100:.3f}%")
print(f"  q50 > 0:     {(results['q50'] > 0).sum()} rows")
print(f"  q50 < 0:     {(results['q50'] < 0).sum()} rows")

# ── RETURNS ───────────────────────────────────────────────────
results['strat_ret'] = results['signal'].shift(1) * results['actual']
results['equity']    = 100_000 * (1 + results['strat_ret']).cumprod()
results['bh_equity'] = 100_000 * (1 + results['actual']).cumprod()

# ── METRICS ───────────────────────────────────────────────────
ann_factor = 252 * 24
ann_ret = results['strat_ret'].mean()  * ann_factor
ann_vol = results['strat_ret'].std()   * np.sqrt(ann_factor)
sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
mdd     = ((results['equity'] - results['equity'].cummax()) /
            results['equity'].cummax()).min()
n_trades = (results['signal'] != 0).sum()

print(f"\n{'═'*45}")
print(f"  CONFIDENCE-BASED BACKTEST")
print(f"{'═'*45}")
print(f"  Ann. Return:     {ann_ret*100:.2f}%")
print(f"  Ann. Volatility: {ann_vol*100:.2f}%")
print(f"  Sharpe Ratio:    {sharpe:.3f}")
print(f"  Max Drawdown:    {mdd*100:.2f}%")
print(f"  Trades taken:    {n_trades} / {n_test} hours")
print(f"  Trade rate:      {n_trades/n_test*100:.1f}%")
print(f"{'═'*45}")

width_threshold = results['width'].quantile(0.40)

# ── VISUALIZATIONS ────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('#0d0d14')
gs  = gridspec.GridSpec(3, 2, figure=fig,
                        hspace=0.45, wspace=0.3)

def style_ax(ax, title):
    ax.set_facecolor('#0d0d14')
    ax.tick_params(colors='#888')
    ax.set_title(title, color='white', fontsize=11)
    ax.grid(True, alpha=0.1)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e1e30')

# ── CHART 1: Quantile prediction fan (first 200 points) ──────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, 'TFT Quantile Predictions — '
              'Shaded band = 80% confidence interval')

n_show = 200
x      = range(n_show)
ax1.fill_between(x, all_q10[:n_show]*100,
                    all_q90[:n_show]*100,
                 alpha=0.25, color='#7b61ff',
                 label='80% interval (q10-q90)')
ax1.plot(x, all_q50[:n_show]*100,
         color='#7b61ff', lw=1.5, label='q50 (median forecast)')
ax1.plot(x, all_targets[:n_show]*100,
         color='white', lw=0.8, alpha=0.7, label='Actual return')
ax1.axhline(0, color='#444', lw=0.8)
ax1.set_xlabel('Test Hours', color='#888')
ax1.set_ylabel('4h Return (%)', color='#888')
ax1.legend(facecolor='#1e1e30', labelcolor='white', fontsize=9)

# ── CHART 2: Equity curve ─────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
style_ax(ax2, f'Confidence-Based Backtest  |  '
              f'Sharpe: {sharpe:.2f}  |  MDD: {mdd*100:.1f}%')
ax2.plot(results.index, results['equity'],
         color='#00ff9d', lw=1.8, label='TFT Strategy')
ax2.plot(results.index, results['bh_equity'],
         color='#7b61ff', lw=1.2, ls='--',
         label='Buy & Hold BTC', alpha=0.8)
ax2.set_ylabel('Portfolio ($)', color='#888')
ax2.legend(facecolor='#1e1e30', labelcolor='white')

# ── CHART 3: Feature Importance (VSN) ────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
style_ax(ax3, 'Feature Importance (Variable Selection Network)')

# Average VSN weights across all test samples and timesteps
mean_vsn = all_vsn.mean(axis=(0, 1))  # (n_features,)
sorted_idx = np.argsort(mean_vsn)[::-1]
top_n      = 15
top_feats  = [feature_cols[i] for i in sorted_idx[:top_n]]
top_vals   = mean_vsn[sorted_idx[:top_n]]

colors = ['#00ff9d' if 'BTC' in f
          else '#ff3c6e' if 'ETH' in f
          else '#ffcc00'
          for f in top_feats]

bars = ax3.barh(range(top_n), top_vals[::-1], color=colors[::-1])
ax3.set_yticks(range(top_n))
ax3.set_yticklabels([f[:20] for f in top_feats[::-1]],
                     color='white', fontsize=8)
ax3.set_xlabel('Avg Selection Weight', color='#888')

# Legend
from matplotlib.patches import Patch
ax3.legend(handles=[
    Patch(color='#00ff9d', label='BTC features'),
    Patch(color='#ff3c6e', label='ETH features'),
    Patch(color='#ffcc00', label='Cross-asset'),
], facecolor='#1e1e30', labelcolor='white', fontsize=8)

# ── CHART 4: Interval width over time ────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
style_ax(ax4, 'Prediction Interval Width Over Time\n'
              '(narrow = confident, wide = uncertain)')
ax4.plot(results.index,
         results['width']*100,
         color='#ffcc00', lw=0.8, alpha=0.7)
ax4.axhline(width_threshold*100,
            color='#00ff9d', lw=1.2, ls='--',
            label=f'Trade threshold ({width_threshold*100:.3f}%)')
ax4.fill_between(results.index,
                 results['width']*100,
                 width_threshold*100,
                 where=results['width']*100 < width_threshold*100,
                 alpha=0.3, color='#00ff9d',
                 label='Confident zones (trades)')
ax4.set_ylabel('Interval Width (%)', color='#888')
ax4.legend(facecolor='#1e1e30', labelcolor='white', fontsize=8)

os.makedirs('results', exist_ok=True)
plt.savefig('results/p03_evaluation.png',
            dpi=150, bbox_inches='tight',
            facecolor='#0d0d14')
plt.show()
print("\n[eval] Saved → results/p03_evaluation.png")
print("[eval] P03 COMPLETE ")