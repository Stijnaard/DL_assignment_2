"""
https://arxiv.org/abs/1611.08024
EEGNet uses 2D convolutions to separately learn:
- Temporal patterns: how brain signals change over time
- Spatial patterns: which sensors activate together

Block 1 — Temporal Conv:  learns frequency filters (like bandpass filters)
Block 2 — Depthwise Conv: learns which sensors to combine per temporal filter
Block 3 — Separable Conv: refines and compresses the learned features

- Xavier is designed for symmetric activations (e.g., Tanh, Sigmoid), 
- Kaiming for asymmetrical, piecewise linear activations (e.g., ReLU) 
Pro's: few parameters
"""

import torch
import torch.nn as nn
from src.config.config import *

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()        
        # Block 1: Temporal Convolution
        # Input:  (B, 1, 248, 200)  treated as a single-channel "image"
        # Output: (B, F1, 248, 200) F1 temporal filter responses
        # Learns frequential features
        self.block1 = nn.Sequential(
            nn.Conv2d(1, EEGNET_F1,
                kernel_size = (1, EEGNET_KERNEL_SIZE),
                padding = (0, EEGNET_KERNEL_SIZE // 2),
                bias = False),
            nn.BatchNorm2d(EEGNET_F1))

        # Block 2: Depthwise Spatial Convolution
        # Input:  (B, F1, 248, 200)
        # Output: (B, F2,   1, ~50) collapsed sensor dimension + pooled time
        # Learns which sensors co-activate for each temporal pattern
        self.block2 = nn.Sequential(
            nn.Conv2d(EEGNET_F1, EEGNET_F2,
                kernel_size = (N_CHANNELS, 1),
                groups = EEGNET_F1,             # Depthwise: each filter independently
                bias = False),
            nn.BatchNorm2d(EEGNET_F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size = (1, 4)), # Reduce time dimension by 4x
            nn.Dropout(EEGNET_DROPOUT))

        # Block 3: Separable Convolution
        # Input:  (B, F2, 1, ~50)
        # Output: (B, F2, 1, ~6)  compresed
        # Refines temporal features in a compute-efficient way
        self.block3 = nn.Sequential(
            nn.Conv2d(EEGNET_F2, EEGNET_F2, kernel_size = (1, 16), padding = (0, 8),
                groups = EEGNET_F2, bias = False), # Depthwise temporal
            nn.Conv2d(EEGNET_F2, EEGNET_F2, kernel_size = (1, 1), bias = False), # Pointwise mix
            nn.BatchNorm2d(EEGNET_F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size = (1, 8)),
            nn.Dropout(EEGNET_DROPOUT))
        
        flat = self.flatting_size()
        self.head = nn.Linear(flat, NUM_CLASSES) # Classifier
        self.init_weights()

    def flatting_size(self) -> int:
        """
        Calculate the number of features output by block3,
        so we can set the input size of the final linear layer
        """
        with torch.no_grad():
            temp = torch.zeros(1, 1, N_CHANNELS, WINDOW_SIZE)
            out = self.block3(self.block2(self.block1(temp)))
        return out.numel()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity = "relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (batch, 248, 200) -> logits: (batch, 4)"""
        x = x.unsqueeze(1)   # (B, 1, 248, 200) add channel dim for Conv2d
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)     # (B, flat)
        return self.head(x)
