from torch import all

from dl_assignment_2.modeling.model import SimpleRNN

from torch import Tensor, ones, tensor, all

class Test_output_shapes:
    rnn = SimpleRNN(10,20,4)
    def test_one(self):
        x: Tensor = ones((3, 10))   # BATCH, SEQ LEN, INPUT DIM
        y = self.rnn(x)
        assert y.shape == (1,4)
        
    def test_batch(self):
        x: Tensor = ones(5,3, 10)
        y = self.rnn(x)
        assert y.shape == (1, 5, 4) # always adds extra dimension at axis=0
        
class Test_Predicion_Sanity:
    rnn = SimpleRNN(10,20,4)
    
    def test_one(self):
        x: Tensor = ones((3, 10))   # BATCH, SEQ LEN, INPUT DIM
        y = self.rnn.predict(x)
        logits = self.rnn(x)
        
    def test_batch(self):
        x: Tensor = ones((2, 3, 10))   # BATCH, SEQ LEN, INPUT DIM
        y = self.rnn.predict(x)
        logits = self.rnn(x)
        
class Test_loss_computation:
    pass