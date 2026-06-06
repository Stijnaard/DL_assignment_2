"""
TemporalAttention: Patch-based Transformer for MEG brain-state classification.

This model is architecturally distinct from every other model in the project:
  - No CNN frontend (unlike CNNTransformer)
  - No recurrence (unlike LSTM / GRU / RNN)
  - No 2D convolutions (unlike EEGNet)

Inspiration: Vision Transformer (ViT, Dosovitskiy 2020) applied to 1D time-series.

How it works:
  1. Patch embedding
       The 200-step signal is divided into non-overlapping patches of width P.
       Each patch (248 sensors × P time-steps) is linearly projected to d_model.
       This gives T/P tokens   e.g. P=10 -> 20 tokens   a much shorter sequence
       than feeding raw time-steps to the Transformer.
       Shorter sequences mean quadratic attention cost stays small.

  2. Sensor mixing (depthwise)
       Before patching, a lightweight depthwise Conv1d mixes information across
       adjacent time-steps within each sensor channel, acting as a local feature
       extractor without cross-channel mixing (that happens implicitly via the
       patch projection).

  3. Learned CLS token
       A learnable [CLS] token is prepended to the patch sequence, same as BERT.
       After the Transformer encoder, only the CLS token's output is used for
       classification   it aggregates global context from all patches.

  4. Transformer encoder
       Standard multi-head self-attention + feedforward, with Pre-LN (more
       stable than Post-LN for small datasets), GELU activations.

  5. Classification head
       CLS output -> LayerNorm -> Dropout -> Linear(4)

Why this suits MEG:
  - Brain state transitions happen over 100–500 ms windows; patching at 50 ms
    (P=10 at 200 Hz) captures the relevant temporal granularity.
  - Self-attention can model long-range dependencies between patches (e.g.
    early sensory response gating later cognitive activity) without the
    vanishing-gradient problems of RNNs.
  - Parameter-efficient: ~190K params vs 724K for CNNTransformer.

Config keys (add to config.py   defaults used if absent):
  TEMPORAL_PATCH_SIZE = 10      # time-steps per patch (200 / 10 = 20 tokens)
  TEMPORAL_D_MODEL    = 128
  TEMPORAL_NHEAD      = 4
  TEMPORAL_LAYERS     = 4
  TEMPORAL_DIM_FF     = 256
  TEMPORAL_DROPOUT    = 0.2
"""

import math
import torch
import torch.nn as nn

from src.config.config import *

class PatchEmbedding(nn.Module):
    """
    Split (B, C, T) into non-overlapping patches of width P, then project each
    patch to d_model.

    Steps:
      - Depthwise Conv1d with kernel=P, stride=P extracts local features within
        each sensor channel (no cross-channel mixing yet).
      - Flatten patch dimension: (B, C, P) -> (B, C*P)  per token.
      - Linear projection: C*P -> d_model.
    """
    def __init__(self, n_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size

        # Depthwise temporal smoothing before patching
        self.pre_conv = nn.Sequential(
            nn.Conv1d(n_channels, n_channels,
                      kernel_size = 3, padding = 1,
                      groups = n_channels, bias = False),   # depthwise
            nn.BatchNorm1d(n_channels),
            nn.GELU())

        # Linear patch projection: (C * P) -> d_model
        self.proj = nn.Linear(n_channels * patch_size, d_model, bias = False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)
        out: (B, n_patches, d_model)
        """
        B, C, T = x.shape
        P = self.patch_size
        assert T % P == 0, f"WINDOW_SIZE ({T}) must be divisible by TEMPORAL_PATCH_SIZE ({P})"

        x = self.pre_conv(x)                    # (B, C, T)

        # Reshape into patches: (B, n_patches, C*P)
        n = T // P
        x = x.reshape(B, C, n, P)              # (B, C, n, P)
        x = x.permute(0, 2, 1, 3)              # (B, n, C, P)
        x = x.reshape(B, n, C * P)             # (B, n, C*P)

        return self.norm(self.proj(x))          # (B, n, d_model)


# ── Sinusoidal positional encoding ───────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Full model ────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Patch-based Transformer for MEG classification.

    Input : (batch, 248, 200)
    Output: (batch, 4)
    """
    def __init__(self):
        super().__init__()
        d_model = TEMPORAL_D_MODEL
        dropout = TEMPORAL_DROPOUT

        # 1. Patch embedding
        self.patch_embed = PatchEmbedding(N_CHANNELS, TEMPORAL_PATCH_SIZE, d_model)

        # 2. Learnable CLS token   prepended before positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std = 0.02)

        # 3. Positional encoding (covers CLS + all patches)
        n_patches = WINDOW_SIZE // TEMPORAL_PATCH_SIZE
        self.pos_enc = PositionalEncoding(d_model, max_len = n_patches + 1,
            dropout = dropout)

        # 4. Transformer encoder (Pre-LN for stable training)
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = TEMPORAL_NHEAD,
            dim_feedforward = TEMPORAL_DIM_FF,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers = TEMPORAL_LAYERS,
            enable_nested_tensor = False)

        # 5. Classification head (uses CLS token only)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, NUM_CLASSES))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity = "relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 248, 200) -> logits: (B, 4)"""
        B = x.size(0)
        # Patch embedding: (B, 248, 200) -> (B, n_patches, d_model)
        tokens = self.patch_embed(x)
        # Prepend CLS token: (B, n_patches+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim = 1)
        # Add positional information
        tokens = self.pos_enc(tokens)
        # Self-attention across all tokens
        tokens = self.transformer(tokens)   # (B, n_patches+1, d_model)
        # Use CLS token for classification
        cls_out = tokens[:, 0]              # (B, d_model)
        return self.head(cls_out)           # (B, 4)