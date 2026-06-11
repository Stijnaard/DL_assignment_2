"""
RNN implementation:
Probably weak due to the fact that RNN:
- forget things that happened many steps ago/vanishing gradient
"""

import torch
import torch.nn as nn

from dl_assignment_2.Niels_models.config import *

class RNNClassifier(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int):
        super().__init__()

        n_channels    = c_in
        hidden        = RNN_HIDDEN
        n_layers      = RNN_LAYERS
        dropout       = RNN_DROPOUT
        bidirectional = RNN_BIDIR

        # 1. Spatial projection
        # Compress N_CHANNELS sensors -> hidden_size features.
        self.input_sequence = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU())

        # 2. Setup RNN
        self.rnn = nn.RNN(
            input_size    = hidden,
            hidden_size   = hidden,
            num_layers    = n_layers,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = bidirectional,
            batch_first   = False # input shape: (time, batch, features)
        )
        out_size = hidden * (2 if bidirectional else 1)

        # 3. Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_size, c_out))
        self.init_weights()

    def init_weights(self):
        """Better weight initialisation -> faster, more stable training."""
        for name, p in self.rnn.named_parameters():
            if   "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias"      in name: nn.init.zeros_(p)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (batch, N_CHANNELS, 200) -> logits: (batch, c_out)"""
        # Reorder axes: (B, C, T) -> (T, B, C)  because RNN expects time-first
        #x = x.transpose(1, 2)
        x = x.permute(2, 0, 1) # (T, B, N_CHANNELS)

        # Project each time step's sensor values
        T, B, C = x.shape
        x = self.input_sequence(x.reshape(T * B, C)).reshape(T, B, -1)   # (T, B, hidden)

        # Run RNN; h_n is the final hidden state
        _, h_n = self.rnn(x)                            # h_n: (layers*dirs, B, hidden)

        # Grab last layer's hidden state from both directions
        if RNN_BIDIR:
            h = torch.cat([h_n[-2], h_n[-1]], dim = -1) # (B, hidden*2)
        else:
            h = h_n[-1]

        return self.head(h) # (B, c_out)
