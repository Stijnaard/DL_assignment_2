import torch
import torch.nn as nn



class MEG1DCNN2(nn.Module):
    def __init__(self, n_channels=248, n_classes=4, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MEG1DCNN3(nn.Module):
    def __init__(self, n_channels=248, n_classes=4, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))






"""
CNN1D-ResNet: 1D Residual CNN for MEG brain-state classification.

The existing CNN1D stacks Conv -> BN -> GELU -> MaxPool blocks without any
skip connections. As the network deepens, gradients shrink through each
non-linearity (vanishing gradient). Residual (skip) connections let the
gradient flow directly back through the shortcut path, making it safe to
add more layers without degradation.

Architecture:
1. Input projection  : Conv1d(N_CHANNELS -> CNN1D_CHANNELS[0], k = 1) learned spatial mixing
2. Residual blocks   : 3 x ResBlock(CNN1D_CHANNELS[0] -> CNN1D_CHANNELS[1], 
                        CNN1D_CHANNELS[1] -> CNN1D_CHANNELS[2], CNN1D_CHANNELS[2] -> CNN1D_CHANNELS[2])
    Each block: Conv -> BN -> GELU -> Dropout -> Conv -> BN + shortcut (1x1 if ch changes)
    Followed by MaxPool(2) to halve the time dimension
3. Global average pool -> flatten
4. Classification head: Linear -> GELU -> Dropout -> Linear(4)

Compared to CNN1D the extra depth + skip connections allow richer feature
hierarchies while keeping training stable.
"""

class ResBlock1D(nn.Module):
    """
    Pre-activation residual block for 1D signals.

    Two Conv1d layers with BN + GELU, plus a shortcut connection.
    If in_ch != out_ch a 1x1 conv aligns dimensions for the shortcut.
    MaxPool(2) after the addition halves the time dimension.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float):
        super().__init__()
        pad = kernel // 2

        self.conv_path = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding = pad, bias = False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding = pad, bias = False),
            nn.BatchNorm1d(out_ch))

        # Shortcut: 1×1 conv only when channel count changes
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size = 1, bias = False),
                nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity())

        self.act  = nn.GELU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_path(x) + self.shortcut(x)
        return self.pool(self.act(out))

class CNN1DResNet(nn.Module):
    """
    1D ResNet classifier.

    Input : (batch, N_CHANNELS, 200)
    Output: (batch, 4)
    """
    def __init__(self, c_in: int, c_out: int, seq_len: int):
        super().__init__()
        channels = CNN1D_RN_CHANNELS
        kernel   = CNN1D_RN_KERNEL
        dropout  = CNN1D_RN_DROPOUT

        # 1. Spatial projection: mix N_CHANNELS sensors into the first channel count
        self.input_proj = nn.Sequential(
            nn.Conv1d(c_in, channels[0], kernel_size = 1, bias = False),
            nn.BatchNorm1d(channels[0]),
            nn.GELU())

        # 2. Stack of residual blocks
        # Each block transitions between consecutive channel sizes and
        # halves the time dimension via MaxPool.
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(ResBlock1D(channels[i], channels[i + 1], kernel, dropout))
        # Extra same-channel residual block at the deepest level for more capacity
        blocks.append(ResBlock1D(channels[-1], channels[-1], kernel, dropout))
        self.res_blocks = nn.Sequential(*blocks)

        # 3. Global average pool -> one vector per channel
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 4. Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(channels[-1]),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(channels[-1] // 2, c_out))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity = "relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, c_in, seq_len) -> logits: (B, c_out)"""
        #x = x.transpose(1, 2)    # (B, c_in, seq_len) -> (B, seq_len, c_in)
        x = self.input_proj(x)   # (B, 64,  200)
        x = self.res_blocks(x)   # (B, 256,  ~25) after pooling
        x = self.gap(x)          # (B, 256,    1)
        return self.head(x)      # (B, c_out)
    

"""
InceptionTime: 1D InceptionTime model for MEG brain-state classification.
"""


from tsai.models.InceptionTime import InceptionTime as _InceptionTime

#InceptionTime = _InceptionTime

class InceptionTime(_InceptionTime):
    def __init__(self, c_in, c_out, seq_len=None, nf=32, nb_filters=None, **kwargs):
        super().__init__(c_in, c_out, seq_len, nf, nb_filters, **kwargs)

    def forward(self, x):
        """x: (batch, timepoints, channels) -> logits: (batch, c_out)"""
        x = x.permute(0, 2, 1) # (N, T, sensors) -> (N, sensors, T)
        return super().forward(x)

__all__ = ["InceptionTime"]