"""
model.py
────────
LSTM with Attention mechanism for crypto direction prediction.

ARCHITECTURE:
  Input (60, 15)
      ↓
  LSTM  ×2 layers    ← learns temporal patterns
      ↓
  Attention Layer    ← learns which timesteps matter most
      ↓
  Dropout            ← prevents overfitting
      ↓
  Linear → 2         ← outputs [P(DOWN), P(UP)]
"""

import torch
import torch.nn as nn
import numpy as np


class AttentionLayer(nn.Module):
    """
    Self-attention over LSTM hidden states.

    For each timestep, compute a score of how important
    that timestep is. Scores sum to 1 (via softmax).
    Then return weighted sum of all hidden states.

    This is the SAME idea as in Transformers — just simpler.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Learnable weight vector — one number per hidden unit
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: shape (batch, seq_len, hidden_size)

        Returns:
            context: shape (batch, hidden_size)  — weighted sum
            scores:  shape (batch, seq_len)       — attention weights
        """
        # Score each timestep: (batch, seq_len, 1)
        scores = self.attention_weights(lstm_output)

        # Remove last dim: (batch, seq_len)
        scores = scores.squeeze(-1)

        # Softmax: scores sum to 1 across time dimension
        # Higher score = this timestep gets more weight
        attn_weights = torch.softmax(scores, dim=1)

        # Weighted sum of all LSTM outputs
        # (batch, seq_len, 1) × (batch, seq_len, hidden) → sum
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output                  # (batch, seq_len, hidden)
        ).squeeze(1)                     # (batch, hidden)

        return context, attn_weights


class CryptoLSTM(nn.Module):
    """
    Full LSTM + Attention model.

    Parameters:
        input_size:   number of features (15)
        hidden_size:  LSTM hidden dimension (tune this)
        num_layers:   stacked LSTM layers (2 is standard)
        dropout:      regularization (prevents memorizing train data)
        num_classes:  2 (UP or DOWN)
    """

    def __init__(self,
                 input_size:  int = 15,
                 hidden_size: int = 128,
                 num_layers:  int = 2,
                 dropout:     float = 0.3,
                 num_classes: int = 2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── LSTM LAYERS ─────────────────────────────────────
        # batch_first=True means input shape is (batch, seq, features)
        # dropout between LSTM layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )

        # ── ATTENTION ───────────────────────────────────────
        self.attention = AttentionLayer(hidden_size)

        # ── CLASSIFIER HEAD ─────────────────────────────────
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)

        Returns:
            logits:       (batch_size, 2)
            attn_weights: (batch_size, seq_len)  for visualization
        """
        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_size) — all timesteps
        # (h_n, c_n): final hidden + cell state
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply layer norm for training stability
        lstm_out = self.layer_norm(lstm_out)

        # Attention: which timesteps matter most?
        context, attn_weights = self.attention(lstm_out)

        # Dropout for regularization
        context = self.dropout(context)

        # Final classification
        logits = self.classifier(context)

        return logits, attn_weights


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters — bigger model = more capacity."""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)