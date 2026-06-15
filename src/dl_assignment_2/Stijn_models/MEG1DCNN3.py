from torch import nn


class MEG1DCNN3(nn.Module):
    def __init__(self, c_in, c_out, seq_len, dropout=0.5):
        """
        1D CNN for MEG classification with 3 convolutional layers and a fully connected layer.
        
        Args:
            c_in (int): Number of input channels (sensors).
            c_out (int): Number of output classes.
            seq_len (int): Length of the input sequence (timepoints).
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(c_in, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16, c_out),
        )

    def forward(self, x):
        """x: (batch, timepoints, channels) -> logits: (batch, c_out)"""

        x = x.permute(0, 2, 1)
        
        return self.classifier(self.features(x))