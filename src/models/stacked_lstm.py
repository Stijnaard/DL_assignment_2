"""
Architecture:
- Bidirectional LSTM to capture both forward and backward temporal dependencies.
- Useful when future context helps interpret the past (e.g. smoothing trends).
- Produces richer hidden states than a unidirectional LSTM at the cost of
roughly 2x the parameters.
"""

import torch
import torch.nn as nn

from src.config.config import *

class AttentionPool(nn.Module):
    """Weighted average of T hidden vectors -> one context vector"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (T, B, H) -> context: (B, H)"""
        w = torch.softmax(self.score(h), dim = 0)  # (T, B, 1)
        return (w * h).sum(dim = 0)                # (B, H)

class StackedLSTM(nn.Module):
    """
    Bidirectional stacked LSTM for 4-class MEG brain-state classification.

    Input : (batch, 248, 200)   — sensors x time
    Output: (batch, 4)          — class logits
    """
    def __init__(self):
        super().__init__()
        hidden    = STACKED_HIDDEN_SIZE
        n_layers  = STACKED_NUM_LAYERS
        dropout   = STACKED_DROPOUT_RATE

        # 1. Spatial projection: compress 248 sensors -> hidden features
        # Applied identically to every time step before the LSTM.
        self.input_proj = nn.Sequential(
            nn.Linear(N_CHANNELS, hidden),
            nn.LayerNorm(hidden),
            nn.GELU())

        # 2. Stacked bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size    = hidden,
            hidden_size   = hidden,
            num_layers    = n_layers,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = True,
            batch_first   = False)
        out_size = hidden * 2 # Bidirectional doubles the output dimension

        # 3. Attention pooling over all T time steps
        self.attention = AttentionPool(out_size)

        # 4. Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(out_size),
            nn.Dropout(dropout),
            nn.Linear(out_size, out_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(out_size // 2, NUM_CLASSES))

        self._init_weights()
        print(f"\nStackedLSTM: hidden = {hidden}, layers = {n_layers}, "
              f"bidirectional = True, params = {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        # LSTM gates: input-hidden Xavier, hidden-hidden orthogonal, biases zero
        for name, p in self.lstm.named_parameters():
            if   "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias"      in name: nn.init.zeros_(p)
        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 248, 200)
        Returns logits : (batch, 4)
        """
        # (B, C, T) -> (T, B, C) time-first for the LSTM
        x = x.permute(2, 0, 1)               # (T, B, 248)
        T, B, C = x.shape
        # Project each time-step's sensor values independently
        x = self.input_proj(x.reshape(T * B, C)).reshape(T, B, -1)  # (T, B, hidden)
        # Run stacked bidir LSTM, out contains all hidden states
        out, _ = self.lstm(x)                 # (T, B, hidden*2)
        # Attention-weighted pooling -> single vector per sample
        context = self.attention(out)         # (B, hidden*2)

        return self.head(context)             # (B, 4)
