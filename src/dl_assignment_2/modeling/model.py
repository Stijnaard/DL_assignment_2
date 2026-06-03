from torch import nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, input_dim: int, hidden_dim: int):
        pass

    def forward(self, x):
        pass

    