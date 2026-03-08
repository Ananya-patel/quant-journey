"""
model.py — Temporal Fusion Transformer (TFT)
─────────────────────────────────────────────
Based on: "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (Lim et al., 2020)

Key components:
  1. Gated Residual Network (GRN)     — adaptive processing
  2. Variable Selection Network (VSN) — feature importance
  3. LSTM Encoder                     — temporal encoding
  4. Multi-Head Attention             — temporal patterns
  5. Quantile Output                  — uncertainty estimation
"""

import torch
import torch.nn as nn
import numpy as np


# ══════════════════════════════════════════════════════════════
# BUILDING BLOCK 1: Gated Linear Unit (GLU)
# ══════════════════════════════════════════════════════════════

class GLU(nn.Module):
    """
    Gated Linear Unit.
    Splits input in half: one half is the value,
    other half is the gate (sigmoid → 0 to 1).
    Output = value × gate

    If gate ≈ 0 → block this transformation
    If gate ≈ 1 → pass this transformation through
    Network learns when to open/close the gate.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size * 2)

    def forward(self, x):
        x  = self.linear(x)
        # Split into two halves along last dim
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


# ══════════════════════════════════════════════════════════════
# BUILDING BLOCK 2: Gated Residual Network (GRN)
# ══════════════════════════════════════════════════════════════

class GRN(nn.Module):
    """
    Gated Residual Network — core TFT building block.

    Architecture:
      input → Linear → ELU → Linear → GLU → + input → LayerNorm

    The skip connection means: if transformation isn't useful,
    the network can learn to ignore it (gate closes).
    """
    def __init__(self, input_size: int,
                 hidden_size: int,
                 output_size: int = None,
                 dropout: float = 0.1):
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.fc1      = nn.Linear(input_size, hidden_size)
        self.elu      = nn.ELU()
        self.fc2      = nn.Linear(hidden_size, output_size)
        self.glu      = GLU(output_size)
        self.dropout  = nn.Dropout(dropout)
        self.layernorm= nn.LayerNorm(output_size)

        # Skip connection projection (if sizes differ)
        if input_size != output_size:
            self.skip_proj = nn.Linear(input_size, output_size,
                                       bias=False)
        else:
            self.skip_proj = None

    def forward(self, x):
        # Skip connection
        skip = x if self.skip_proj is None else self.skip_proj(x)

        # Main path
        out = self.fc1(x)
        out = self.elu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.glu(out)

        # Add & Norm
        return self.layernorm(out + skip)


# ══════════════════════════════════════════════════════════════
# BUILDING BLOCK 3: Variable Selection Network (VSN)
# ══════════════════════════════════════════════════════════════

class VariableSelectionNetwork(nn.Module):
    """
    Learns which features matter most at each timestep.

    For each feature:
      - Apply a GRN to get a processed representation
    Then:
      - Compute softmax weights over all features
      - Return weighted sum

    Output: (batch, seq_len, hidden_size)
            + weights: (batch, seq_len, n_features)
                       ↑ interpretable feature importance!
    """
    def __init__(self, n_features: int,
                 hidden_size: int,
                 dropout: float = 0.1):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size

        # One GRN per feature (processes each feature independently)
        self.feature_grns = nn.ModuleList([
            GRN(1, hidden_size, hidden_size, dropout)
            for _ in range(n_features)
        ])

        # Softmax weight generator
        # Takes all features concatenated → outputs n_features weights
        self.weight_grn = GRN(n_features, hidden_size,
                              n_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            out:     (batch, seq_len, hidden_size)
            weights: (batch, seq_len, n_features)
        """
        # Process each feature independently
        processed = []
        for i, grn in enumerate(self.feature_grns):
            # Take feature i: (batch, seq_len, 1)
            feat = x[:, :, i:i+1]
            processed.append(grn(feat))  # (batch, seq_len, hidden)

        # Stack: (batch, seq_len, n_features, hidden)
        processed = torch.stack(processed, dim=2)

        # Compute selection weights from raw features
        weights = self.weight_grn(x)          # (batch, seq, n_feat)
        weights = self.softmax(weights)        # sum to 1

        # Weighted sum across features
        # weights: (batch, seq, n_feat, 1)
        # processed: (batch, seq, n_feat, hidden)
        out = (weights.unsqueeze(-1) * processed).sum(dim=2)

        return out, weights


# ══════════════════════════════════════════════════════════════
# BUILDING BLOCK 4: Multi-Head Attention (simplified)
# ══════════════════════════════════════════════════════════════

class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention where all heads share value weights.
    This makes it interpretable — we can average attention
    across heads and get meaningful temporal weights.
    """
    def __init__(self, hidden_size: int,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.n_heads    = n_heads
        self.head_size  = hidden_size // n_heads
        self.hidden_size= hidden_size

        # Query and Key projections (one per head)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        # Shared value projection (interpretability)
        self.W_v = nn.Linear(hidden_size, self.head_size)
        self.W_o = nn.Linear(self.head_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale   = self.head_size ** -0.5

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            out:     (batch, seq_len, hidden_size)
            attn:    (batch, seq_len, seq_len) — averaged over heads
        """
        B, T, H = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.head_size)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_size)
        V = self.W_v(x)  # shared: (B, T, head_size)

        # Attention scores for each head
        Q = Q.transpose(1, 2)  # (B, heads, T, head_size)
        K = K.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) * self.scale
        attn   = torch.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        # Average attention across heads for interpretability
        attn_avg = attn.mean(dim=1)  # (B, T, T)

        # Apply attention to shared values
        # Use averaged attention weights
        out = torch.matmul(attn_avg, V)  # (B, T, head_size)
        out = self.W_o(out)              # (B, T, hidden_size)

        return out, attn_avg


# ══════════════════════════════════════════════════════════════
# FULL TFT MODEL
# ══════════════════════════════════════════════════════════════

class TemporalFusionTransformer(nn.Module):
    """
    Full TFT for quantile regression.

    Predicts 3 quantiles: q10, q50, q90
    These represent: worst case, median, best case
    """

    def __init__(self,
                 n_features:  int,
                 hidden_size: int   = 64,
                 lstm_layers: int   = 2,
                 n_heads:     int   = 4,
                 dropout:     float = 0.1,
                 quantiles:   list  = [0.1, 0.5, 0.9]):
        super().__init__()

        self.quantiles   = quantiles
        self.n_quantiles = len(quantiles)
        self.hidden_size = hidden_size

        # 1. Variable Selection
        self.vsn = VariableSelectionNetwork(
                        n_features, hidden_size, dropout)

        # 2. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0
        )

        # 3. Post-LSTM GRN
        self.post_lstm_grn = GRN(hidden_size, hidden_size,
                                 dropout=dropout)

        # 4. Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(
                            hidden_size, n_heads, dropout)

        # 5. Post-attention GRN
        self.post_attn_grn = GRN(hidden_size, hidden_size,
                                 dropout=dropout)

        # 6. Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 7. Quantile output heads (one per quantile)
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in quantiles
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            quantile_preds: (batch, n_quantiles)
            vsn_weights:    (batch, seq_len, n_features)
            attn_weights:   (batch, seq_len, seq_len)
        """
        # 1. Variable selection
        x_selected, vsn_weights = self.vsn(x)

        # 2. LSTM encoding
        lstm_out, _ = self.lstm(x_selected)

        # 3. Post-LSTM GRN
        lstm_out = self.post_lstm_grn(lstm_out)

        # 4. Multi-head attention
        attn_out, attn_weights = self.attention(lstm_out)

        # 5. Post-attention GRN + residual
        out = self.post_attn_grn(attn_out)
        out = self.layer_norm(out + lstm_out)  # residual

        # 6. Use last timestep for prediction
        last = out[:, -1, :]  # (batch, hidden_size)

        # 7. Predict each quantile
        quantile_preds = torch.cat([
            head(last) for head in self.quantile_heads
        ], dim=-1)  # (batch, n_quantiles)

        return quantile_preds, vsn_weights, attn_weights


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)