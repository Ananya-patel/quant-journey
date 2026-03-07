"""
evaluate.py
───────────
Final evaluation on the HELD-OUT test set.
This is the only honest measure of model performance.
Run this ONCE — after this, no more model changes.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, pickle, json

from src.data_loader import load_data
from src.features    import build_features
from src.dataset     import prepare_dataloaders
from src.model       import CryptoLSTM

# ── CONFIG (must match main.py exactly) ───────────────────────
SEQ_LEN     = 60
BATCH_SIZE  = 64
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.35

# ── LOAD EVERYTHING ───────────────────────────────────────────
print("\n" + "═"*55)
print("  P02 — Final Test Set Evaluation")
print("═"*55)

df_raw = load_data("btc_1h.csv")
df, feature_cols = build_features(df_raw)

_, _, test_loader, scaler, info = prepare_dataloaders(
    df, feature_cols, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

# Load best saved model
model = CryptoLSTM(input_size=len(feature_cols),
                   hidden_size=HIDDEN_SIZE,
                   num_layers=NUM_LAYERS,
                   dropout=DROPOUT)

model_path = os.path.join("models", "best_model.pt")
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
print(f"\n[eval] Loaded model from {model_path}")

# ── INFERENCE ON TEST SET ─────────────────────────────────────
all_preds   = []
all_labels  = []
all_probs   = []
all_attns   = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits, attn = model(X_batch)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())
        all_probs.extend(probs[:, 1].numpy())  # P(UP)
        all_attns.extend(attn.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)
all_attns  = np.array(all_attns)   # shape: (n_samples, seq_len)

# ── METRICS ───────────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, roc_auc_score)

acc       = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall    = recall_score(all_labels, all_preds, zero_division=0)
f1        = f1_score(all_labels, all_preds, zero_division=0)
auc       = roc_auc_score(all_labels, all_probs)
cm        = confusion_matrix(all_labels, all_preds)

print(f"\n{'═'*40}")
print(f"  TEST SET RESULTS")
print(f"{'═'*40}")
print(f"  Accuracy:       {acc*100:.2f}%")
print(f"  Precision:      {precision*100:.2f}%")
print(f"  Recall:         {recall*100:.2f}%")
print(f"  F1 Score:       {f1:.4f}")
print(f"  ROC-AUC:        {auc:.4f}")
print(f"{'═'*40}")
print(f"\n  Confusion Matrix:")
print(f"  Predicted →     DOWN    UP")
print(f"  Actual DOWN  [{cm[0,0]:>6}  {cm[0,1]:>6}]")
print(f"  Actual UP    [{cm[1,0]:>6}  {cm[1,1]:>6}]")

# ── SIMPLE BACKTEST FROM MODEL PREDICTIONS ────────────────────
# Get the test portion of prices
n_total  = len(df)
val_end  = int(n_total * 0.85)
test_df  = df.iloc[val_end + SEQ_LEN:].copy()
min_len  = min(len(test_df), len(all_probs))
test_df  = test_df.iloc[:min_len].copy()

test_df['pred']       = all_preds[:min_len]
test_df['prob_up']    = all_probs[:min_len]

# Signal: go long when model says UP, short when DOWN
# Use confidence threshold: only trade when model is sure
CONFIDENCE = 0.53   # only trade when P(UP) > 53% or < 47%

test_df['signal'] = 0
test_df.loc[test_df['prob_up'] >  CONFIDENCE, 'signal'] = 1
test_df.loc[test_df['prob_up'] < (1-CONFIDENCE), 'signal'] = -1

# Strategy return (shift by 1 — no lookahead)
test_df['strat_ret'] = test_df['signal'].shift(1) * test_df['return_1h']
test_df['equity']    = 100_000 * (1 + test_df['strat_ret']).cumprod()
test_df['bh_equity'] = 100_000 * (1 + test_df['return_1h']).cumprod()

# Strategy metrics
ann_ret = test_df['strat_ret'].mean() * 252 * 24  # hourly → annual
ann_vol = test_df['strat_ret'].std()  * np.sqrt(252 * 24)
sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
mdd     = ((test_df['equity'] - test_df['equity'].cummax())
           / test_df['equity'].cummax()).min()

print(f"\n{'═'*40}")
print(f"  BACKTEST (test period only)")
print(f"{'═'*40}")
print(f"  Ann. Return:    {ann_ret*100:.2f}%")
print(f"  Ann. Volatility:{ann_vol*100:.2f}%")
print(f"  Sharpe Ratio:   {sharpe:.3f}")
print(f"  Max Drawdown:   {mdd*100:.2f}%")
print(f"  Trades taken:   {(test_df['signal']!=0).sum()}")
print(f"{'═'*40}")

# ── VISUALIZATIONS ────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#0d0d14')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

style = dict(facecolor='#0d0d14', labelcolor='white')

# 1. Equity curve
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#0d0d14'); ax1.tick_params(colors='white')
ax1.plot(test_df.index, test_df['equity'],
         color='#00ff9d', lw=1.8, label='LSTM Strategy')
ax1.plot(test_df.index, test_df['bh_equity'],
         color='#7b61ff', lw=1.2, ls='--', label='Buy & Hold BTC')
ax1.set_title(f'P02 — Test Period Equity Curve  |  '
              f'Sharpe: {sharpe:.2f}  |  MDD: {mdd*100:.1f}%',
              color='white', fontsize=12)
ax1.legend(facecolor='#1e1e30', labelcolor='white')
ax1.grid(True, alpha=0.1)

# 2. Prediction probability distribution
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#0d0d14'); ax2.tick_params(colors='white')
ax2.hist(all_probs[all_labels==1], bins=30,
         alpha=0.6, color='#00ff9d', label='True UP')
ax2.hist(all_probs[all_labels==0], bins=30,
         alpha=0.6, color='#ff3c6e', label='True DOWN')
ax2.axvline(0.5, color='white', lw=1, ls='--')
ax2.set_title('P(UP) Distribution by True Label', color='white')
ax2.legend(facecolor='#1e1e30', labelcolor='white')
ax2.grid(True, alpha=0.1)

# 3. Confusion matrix heatmap
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#0d0d14'); ax3.tick_params(colors='white')
im = ax3.imshow(cm, cmap='Greens')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Pred DOWN','Pred UP'], color='white')
ax3.set_yticklabels(['True DOWN','True UP'], color='white')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i,j]),
                 ha='center', va='center',
                 color='white', fontsize=16, fontweight='bold')
ax3.set_title('Confusion Matrix', color='white')

# 4. Average attention weights across test set
ax4 = fig.add_subplot(gs[2, :])
ax4.set_facecolor('#0d0d14'); ax4.tick_params(colors='white')
mean_attn = all_attns.mean(axis=0)   # average over all test samples
hours_ago = list(range(SEQ_LEN, 0, -1))
ax4.bar(hours_ago, mean_attn,
        color=['#00ff9d' if a > mean_attn.mean() else '#333'
               for a in mean_attn])
ax4.set_xlabel('Hours Ago', color='white')
ax4.set_ylabel('Attention Weight', color='white')
ax4.set_title(
    'Attention Weights — Which hours does the model focus on?',
    color='white')
ax4.invert_xaxis()
ax4.grid(True, alpha=0.1)

# Annotate top-3 attention hours
top3 = np.argsort(mean_attn)[-3:]
for idx in top3:
    ax4.annotate(f'{hours_ago[idx]}h ago',
                xy=(hours_ago[idx], mean_attn[idx]),
                xytext=(0, 8), textcoords='offset points',
                ha='center', color='#ffcc00', fontsize=9)

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, 'p02_evaluation.png'),
            dpi=150, bbox_inches='tight', facecolor='#0d0d14')
plt.show()
print("\n[eval] Saved → results/p02_evaluation.png")
print("\n[eval] P02 COMPLETE ")