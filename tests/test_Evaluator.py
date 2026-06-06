from dl_assignment_2.modeling.evaluation import Evaluator
from dl_assignment_2.modeling.model import SimpleRNN
from dl_assignment_2.modeling.dataset import TestDataset
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from numpy import random

class Test_stuff:
    x: Tensor = tensor(range(1,16)).reshape(5,3)
    y: Tensor = tensor([1,2,3])
    dl = DataLoader(TestDataset(x, y))
    
    rnn = SimpleRNN(5, 10, 4)
    
    eval = Evaluator(rnn, dl)
    
    def test_forward(self):
        pass