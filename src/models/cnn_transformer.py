"""
CNN Frontend + Transformer Encoder Hybrid
Theoretically the best, since
1. CNN: local pattern extraction (oscillations, spikes)
2. Transformer encoder: long-range dependencies
─ Self-attention: each time step can directly attend to any other step
─ Captures correlations between distant brain events
3. Attention pooling + classification head
─ Weighted average over time steps (similar to GRU)
"""

import math
import torch
import torch.nn as nn
from src.config.config import *

# 1. CNN frontend
class CNNFrontend(nn.Module):
    """
    1D CNN that:
        (a) compresses 248 sensor channels -> d_model features
        (b) reduces 200 time steps -> ~50 steps (less for Transformer to process)
        (c) denoises the signal (local averaging via conv)

    Each Conv1DBlock: Conv -> BN -> GELU -> Dropout
    Each MaxPool halves the time dimension
    """
    def __init__(self, in_channels, cnn_channels, d_model, kernel = 7, dropout = 0.2):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for ch in cnn_channels:
            layers += [
                nn.Conv1d(prev_ch, ch, kernel_size = kernel,
                    padding = kernel // 2, bias = False),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2)]
            prev_ch = ch

        # Final projection to the Transformer's d_model dimension
        layers += [
            nn.Conv1d(prev_ch, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, 248, 200) -> (B, d_model, T_reduced)"""
        return self.net(x)

# Component 2: Positional Encoding
class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal position information to each time step
    The Transformer treats all positions equally, pos. encoding
    add the location, information using sine/cosine waves of different frequencies.

    (Vaswani 2017): https://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Precompute the positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))           # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div) # Even indices -> sine
        pe[:, 1::2] = torch.cos(pos * div) # Odd  indices -> cosine
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer("pe", pe) # Not a learnable parameter
        self.pe : torch.Tensor # pylance error prevention

    def forward(self, x):
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)] # Add position info
        return self.dropout(x)
    
# Component 3: Attention Pooling
class AttentionPool(nn.Module):
    """Weighted average of time steps, same as in GRU model"""
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        """x: (B, T, d_model) -> context: (B, d_model)"""
        w = torch.softmax(self.score(x), dim = 1) # (B, T, 1)
        return (w * x).sum(dim = 1)               # (B, d_model)

# Full Hybrid Model
class CNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model  = CNNTRANS_D_MODEL
        nhead    = CNNTRANS_NHEAD
        n_layers = CNNTRANS_LAYERS
        dim_ff   = CNNTRANS_DIM_FF
        dropout  = CNNTRANS_DROPOUT

        # 1. CNN frontend
        self.cnn = CNNFrontend(
            in_channels  = N_CHANNELS,
            cnn_channels = CNNTRANS_CNN_CHANNELS,
            d_model = d_model,
            dropout = dropout)

        # 2. Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout = dropout)

        # 3. Transformer encoder
        # Multi-head self-attention + feedforward MLP + LayerNorm + residuals
        encoder_layer = nn.TransformerEncoderLayer(
            d_model     = d_model,
            nhead       = nhead,
            dim_feedforward = dim_ff,
            dropout     = dropout,
            activation  = "gelu",
            batch_first = True,   # Input: (batch, seq, features)
            norm_first  = True)   # Pre-LN: more stable training than Post-LN
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = n_layers)

        # 4. Attention pooling
        self.pool = AttentionPool(d_model)

        # 5. Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, NUM_CLASSES))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity = "relu")
            elif isinstance(m, (nn.BatchNorm1d,)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (batch, 248, 200) -> logits: (batch, 4)"""
        x = self.cnn(x)         # CNN: (B, 248, 200) → (B, d_model, T_short)
        x = x.permute(0, 2, 1)  # Reorder: (B, d_model, T) → (B, T, d_model)  for Transformer
        x = self.pos_enc(x)     # Add positional encoding
        # Self-attention across time steps
        x = self.transformer(x) # (B, T, d_model)
        # Weighted pooling -> single vector
        x = self.pool(x)        # (B, d_model)
        return self.head(x)     # (B, 4)
