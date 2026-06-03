from abc import ABC, abstractmethod

from torch.nn import RNN, Linear, Softmax
from torch import nn, Tensor

class BaseModel(nn.Module, ABC):
    singular_sigmoid: Softmax = Softmax(dim=1)
    batch_sigmoid: Softmax = Softmax(dim=2)
    
    @abstractmethod
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x):
        pass
    
    def predict(self, X: Tensor) -> Tensor:
        logits: Tensor = self(X)
        y = self.get_sigmoid(X, logits)
        return y

    def get_sigmoid(self, X, logits):
        if len(X.shape) == 2:
            y: Tensor = self.singular_sigmoid(logits)
        elif len(X.shape) == 3:
            y: Tensor = self.batch_sigmoid(logits)
        else:
            raise ValueError("X's dimensionality exceeds model")
        return y
        
class SimpleRNN(BaseModel):
    def __init__(self, in_dim, hidden_dim: int, out_dim: int):
        super().__init__()
        
        self.rnn: RNN = RNN(in_dim, hidden_dim, 1, batch_first=True)
        self.linear: Linear = Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        _, a = self.rnn(x)
        b = self.linear(a)
        
        return b
