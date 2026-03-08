"""
trainer.py
──────────
Quantile loss (Pinball loss) training for TFT.

NEW CONCEPT: Quantile Loss
  For q=0.9: penalize under-predictions MORE
  For q=0.1: penalize over-predictions MORE
  For q=0.5: equal penalty (same as MAE)

  loss(q, y, ŷ) = q × max(y-ŷ, 0) + (1-q) × max(ŷ-y, 0)

This forces each head to learn its quantile correctly.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json, os


def quantile_loss(preds: torch.Tensor,
                  targets: torch.Tensor,
                  quantiles: list) -> torch.Tensor:
    """
    Pinball loss for multiple quantiles.

    Args:
        preds:     (batch, n_quantiles)
        targets:   (batch,)
        quantiles: [0.1, 0.5, 0.9]
    """
    targets = targets.unsqueeze(1)  # (batch, 1)
    losses  = []

    for i, q in enumerate(quantiles):
        pred = preds[:, i:i+1]       # (batch, 1)
        err  = targets - pred
        # Asymmetric loss
        loss = torch.max(q * err, (q - 1) * err)
        losses.append(loss.mean())

    return torch.stack(losses).mean()


def train_model(model, train_loader, val_loader,
                quantiles=[0.1, 0.5, 0.9],
                n_epochs=40, lr=1e-3, patience=8):

    device    = torch.device('cuda' if torch.cuda.is_available()
                             else 'cpu')
    print(f"[trainer] Device: {device}")
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=3, factor=0.5)

    history  = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    patience_ctr = 0
    model_dir = os.path.join(os.path.dirname(__file__),
                             '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>10}")
    print("─" * 32)

    for epoch in range(1, n_epochs + 1):

        # ── TRAIN ─────────────────────────────────────────────
        model.train()
        t_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds, _, _ = model(X)
            loss = quantile_loss(preds, y, quantiles)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())

        # ── VALIDATE ──────────────────────────────────────────
        model.eval()
        v_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds, _, _ = model(X)
                loss = quantile_loss(preds, y, quantiles)
                v_losses.append(loss.item())

        t_loss = np.mean(t_losses)
        v_loss = np.mean(v_losses)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)

        print(f"{epoch:>5}   {t_loss:>11.6f}   {v_loss:>9.6f}")
        scheduler.step(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            patience_ctr = 0
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'best_tft.pt'))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n[trainer] Early stop at epoch {epoch}")
                break

    with open(os.path.join(model_dir, 'tft_history.json'), 'w') as f:
        json.dump(history, f)

    print(f"\n[trainer] Best val loss: {best_val:.6f}")
    return history


def plot_training(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d0d14')
    ax.set_facecolor('#0d0d14')
    ax.tick_params(colors='white')
    ax.title.set_color('white')

    e = range(1, len(history['train_loss']) + 1)
    ax.plot(e, history['train_loss'],
            color='#00ff9d', lw=2, label='Train')
    ax.plot(e, history['val_loss'],
            color='#ff3c6e', lw=2, label='Val')
    ax.set_title('TFT — Quantile Loss Curves')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Pinball Loss', color='white')
    ax.legend(facecolor='#1e1e30', labelcolor='white')
    ax.grid(True, alpha=0.15)

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/tft_training.png',
                dpi=150, bbox_inches='tight',
                facecolor='#0d0d14')
    plt.show()