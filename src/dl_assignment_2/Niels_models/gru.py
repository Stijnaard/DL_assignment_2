"""
GRU (Gated Recurrent Unit) improves on vanilla RNN by adding two gates:
- Reset gate: decides how much of the past to forget
- Update gate: decides how much new information to add
This solves the vanishing-gradient problem, making GRU probably better than RNN
at remembering patterns that span many time steps.

Attention pooling: instead of only using the final hidden state,
we compute a weighted average of all hidden states. The weights are
learned so the model focuses on the most informative time points.
"""

import torch
import torch.nn as nn

from dl_assignment_2.Niels_models.config import *

class AttentionPool(nn.Module):
    """
    Given T hidden vectors, compute one context vector by taking a
    weighted average where the weights are learned by a small network.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1) # One score per time step

    def forward(self, h):
        """h: (T, B, H) -> context: (B, H)"""
        # Score each time step
        weights = torch.softmax(self.score(h), dim = 0) # (T, B, 1), sum to 1 over T
        # Weighted sum across time
        return (weights * h).sum(dim = 0)               # (B, H)

class GRUClassifier(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int):
        super().__init__()
        n_channels    = c_in
        hidden        = GRU_HIDDEN
        n_layers      = GRU_LAYERS
        dropout       = GRU_DROPOUT
        bidirectional = GRU_BIDIR

        # 1. Spatial projection
        self.input_sequential = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU())

        # 2. Setup GRU
        self.gru = nn.GRU(
            input_size    = hidden,
            hidden_size   = hidden,
            num_layers    = n_layers,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = bidirectional,
            batch_first   = False)
        out_size = hidden * (2 if bidirectional else 1)

        # 3. Attention pooling over all time steps
        self.attention = AttentionPool(out_size)

        # 4. Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(out_size),
            nn.Dropout(dropout),
            nn.Linear(out_size, out_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(out_size // 2, c_out))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.gru.named_parameters():
            if   "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias"      in name: nn.init.zeros_(p)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B, T, C) -> logits: (B, c_out)"""
        
        x = x.permute(1, 0, 2) # (T, B, C)
        
        T, B, C = x.shape
        x = self.input_sequential(x.reshape(T * B, C)).reshape(T, B, -1) # (T, B, hidden)
        out, _ = self.gru(x)                                             # (T, B, out_size)
        context = self.attention(out)                                    # (B, out_size)
        return self.head(context)                                        # (B, c_out)
