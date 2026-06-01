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

class StackedLSTM(nn.Module):
    """
    Bidirectional LSTM model to predict sequence.
    Reads the sequence forwards and backwards, then combines both directions
    before the final linear output.
    """
    def __init__(self,
            input_size:   int   = WINDOW_SIZE,
            hidden_size:  int   = STACKED_HIDDEN_SIZE,
            num_layers:   int   = STACKED_NUM_LAYERS,
            dropout_rate: float = STACKED_DROPOUT_RATE):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout_rate,
            bidirectional = True)

        # Dropout applied after the LSTM output, before the linear layer
        self.dropout = nn.Dropout(p = dropout_rate)

        # If bidirectional=True, the output has hidden_size * 2 features
        # (forward direction + backward direction concatenated)
        self.fc_out = nn.Linear(hidden_size * 2, WINDOW_SIZE)

        print(f"\nStacked MModel created with hidden = {hidden_size} "
            f"and layers = {num_layers} (bidirectional)")

    def forward(self, x):
        """
        Hidden states initialised to zero, last timestep output used for prediction.
        """
        batch_size = x.size(0)

        # num_layers * 2 because bidirectional doubles the number of hidden states
        h0 = torch.zeros(self.num_layers * 2,
            batch_size,
            self.hidden_size,
            device = x.device)
        c0 = torch.zeros(self.num_layers * 2,
            batch_size,
            self.hidden_size,
            device = x.device)

        # Forward pass; lstm_out has shape (batch, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the last timestep output as the sequence summary
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc_out(out).squeeze(-1)

        return out
