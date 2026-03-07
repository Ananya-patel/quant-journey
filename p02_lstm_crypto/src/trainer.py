"""
trainer.py
──────────
Training loop for the LSTM model.

KEY CONCEPTS:
  Epoch:     one full pass through all training data
  Batch:     small chunk of data (64 sequences at once)
  Loss:      how wrong the model is (CrossEntropy for classification)
  Optimizer: algorithm that adjusts weights to reduce loss (Adam)
  Scheduler: reduces learning rate when progress stalls
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def train_model(model,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                n_epochs:     int   = 30,
                lr:           float = 1e-3,
                patience:     int   = 7):
    """
    Full training loop with:
    - Adam optimizer
    - ReduceLROnPlateau scheduler
    - Early stopping (stop if val loss doesn't improve)
    - Best model checkpointing

    Args:
        patience: stop after this many epochs with no improvement
    """

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print(f"[trainer] Using device: {device}")
    model = model.to(device)

    # Loss function: CrossEntropyLoss for binary classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer — industry standard for deep learning
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-4)

    # Reduce LR when val_loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    # ── TRACKING ────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    best_val_loss   = float('inf')
    patience_counter = 0
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n[trainer] Starting training — {n_epochs} epochs max")
    print(f"{'Epoch':>5} {'Train Loss':>11} "
          f"{'Val Loss':>9} {'Train Acc':>10} {'Val Acc':>8}")
    print("─" * 52)

    for epoch in range(1, n_epochs + 1):

        # ── TRAINING PHASE ──────────────────────────────────
        model.train()
        train_losses, train_correct, train_total = [], 0, 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()              # clear gradients
            logits, _ = model(X_batch)         # forward pass
            loss = criterion(logits, y_batch)  # compute loss
            loss.backward()                    # backpropagation

            # Gradient clipping — prevents exploding gradients
            # common issue with LSTMs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()                   # update weights

            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        # ── VALIDATION PHASE ────────────────────────────────
        model.eval()
        val_losses, val_correct, val_total = [], 0, 0

        with torch.no_grad():   # no gradient needed for eval
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)

                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += len(y_batch)

        # ── EPOCH METRICS ───────────────────────────────────
        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        t_acc  = train_correct / train_total
        v_acc  = val_correct   / val_total

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        print(f"{epoch:>5}   {t_loss:>10.4f}   {v_loss:>8.4f}"
              f"   {t_acc:>9.3f}   {v_acc:>7.3f}")

        # ── SCHEDULER ───────────────────────────────────────
        scheduler.step(v_loss)

        # ── EARLY STOPPING + CHECKPOINT ─────────────────────
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[trainer] Early stopping at epoch {epoch}")
                break

    # Save training history
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history, f)

    print(f"\n[trainer] Best val loss: {best_val_loss:.4f}")
    print(f"[trainer] Model saved → models/best_model.pt")

    return history


def plot_training(history: dict) -> None:
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d0d14')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d0d14')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.yaxis.label.set_color('#888')
        ax.xaxis.label.set_color('#888')

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax1.plot(epochs, history['train_loss'],
             color='#00ff9d', lw=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'],
             color='#ff3c6e', lw=2, label='Val Loss')
    ax1.set_title('Loss Curves — watch for overfitting')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CrossEntropy Loss')
    ax1.legend(facecolor='#1e1e30', labelcolor='white')
    ax1.grid(True, alpha=0.15)

    # Accuracy curves
    ax2.plot(epochs, [a*100 for a in history['train_acc']],
             color='#00ff9d', lw=2, label='Train Acc')
    ax2.plot(epochs, [a*100 for a in history['val_acc']],
             color='#ff3c6e', lw=2, label='Val Acc')
    ax2.axhline(50, color='white', lw=0.8,
                linestyle='--', label='Random baseline')
    ax2.set_title('Accuracy — goal: val acc > 52%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(facecolor='#1e1e30', labelcolor='white')
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()

    results_dir = os.path.join(os.path.dirname(__file__),
                               '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight', facecolor='#0d0d14')
    plt.show()
    print("[trainer] Training curves saved → results/")