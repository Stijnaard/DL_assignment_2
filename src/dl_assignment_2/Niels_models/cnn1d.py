"""
A 1D CNN slides small filters along the time axis to detect local patterns.
Each filter learns to recognise a specific short temporal event.

Stacking multiple conv layers lets the network learn hierarchical features:
Layer 1 -> short bursts
Layer 2 -> rhythmic patterns
Layer 3 -> longer events

Pro's: faster to train, local patterns
Con: not so good at very long-range

- Xavier is designed for symmetric activations (e.g., Tanh), 
- Kaiming for asymmetrical, piecewise linear activations (e.g., ReLU)
"""

import torch.nn as nn

from dl_assignment_2.Niels_models.config import *

class Conv1DBlock(nn.Module):
    """One convolutional block: Conv -> BatchNorm -> GELU -> Dropout"""
    def __init__(self, in_ch, out_ch, kernel, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch,
                kernel_size = kernel,
                padding = kernel // 2,
                bias = False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.block(x)

class CNN1DClassifier(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int, dropout: float = CNN1D_DROPOUT):
        super().__init__()
        channels = CNN1D_CHANNELS
        kernel   = CNN1D_KERNEL
        dropout  = dropout

        # 1. Input projection (N_CHANNELS sensors -> first channel count)
        # Learned spatial filter over sensors
        self.input_sequential = nn.Sequential(
            nn.Conv1d(c_in, channels[0], kernel_size = 1, bias = False),
            nn.BatchNorm1d(channels[0]),
            nn.GELU())

        # 2. Stacked conv. blocks with 2x downsampling between layers
        layers = []
        for i in range(len(channels) - 1):
            layers.append(Conv1DBlock(channels[i], channels[i + 1], kernel, dropout))
            layers.append(nn.MaxPool1d(2)) # Downsample time by 2x
        self.conv_blocks = nn.Sequential(*layers)

        # 3. Global average pooling
        # Regardless of time length after conv, we get one vector per filter
        self.gap = nn.AdaptiveAvgPool1d(1) # (B, C, T) -> (B, C, 1)

        # 4. Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Linear(channels[-1] // 2, c_out))
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
        """x: (batch, time, channels) -> logits: (batch, c_out)"""

        x = x.permute(0, 2, 1)          # (B, T, C) -> (B, C, T)
        
        x = self.input_sequential(x) # (B, 64,  200)
        x = self.conv_blocks(x)      # (B, 256,  25)
        x = self.gap(x)              # (B, 256,   1)
        return self.head(x)          # (B, c_out)
