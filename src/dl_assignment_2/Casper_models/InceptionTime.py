# from tsai.models.InceptionTime import InceptionTime as _InceptionTime

# #InceptionTime = _InceptionTime

# class InceptionTime(_InceptionTime):
#     def __init__(self, c_in, c_out, seq_len=None, dropout=0.0, nf=32, nb_filters=None, **kwargs):
#         super().__init__(c_in, c_out, seq_len, nf, nb_filters, **kwargs)

#     def forward(self, x):
#         """x: (batch, timepoints, channels) -> logits: (batch, c_out)"""
#         x = x.permute(0, 2, 1) # (N, T, sensors) -> (N, sensors, T)
#         return super().forward(x)

# __all__ = ["InceptionTime"]

# #test = InceptionTime(c_in=248, c_out=4, seq_len=512)


from tsai.models.InceptionTime import InceptionTime
import torch.nn as nn

class InceptionTimeWithDropout(InceptionTime):
    def __init__(self, c_in, c_out, seq_len=None, dropout=0.0, **kwargs):
        super().__init__(c_in, c_out, seq_len, **kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, timepoints, channels) -> logits: (batch, c_out)"""
        x = x.permute(0, 2, 1) # (N, T, sensors) -> (N, sensors, T)
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x



