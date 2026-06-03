from torch import nn
from abc import ABC, abstractmethod

from torch.nn import RNN, Linear

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x):
        pass
    
class SimpleRNN(BaseModel):
    def __init__(self, in_dim, hidden_dim: int, out_dim: int):
        super().__init__()
        
        self.rnn: RNN = RNN(in_dim, hidden_dim, 1, batch_first=True)
        self.linear: Linear = Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        _, a = self.rnn(x)
        b = self.linear(a)
        
        return b
