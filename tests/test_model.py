from torch import all

from dl_assignment_2.modeling.model import SimpleRNN

from torch import Tensor, ones

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
        assert y.sum(1) == Tensor([1])
        assert not all(logits == y)
        
    def test_batch(self):
        x: Tensor = ones((2, 3, 10))   # BATCH, SEQ LEN, INPUT DIM
        y = self.rnn.predict(x)
        logits = self.rnn(x)
        print(x.shape)
        print(logits)
        print(y)
        assert y.sum(1).item() == 1
        assert y.shape == (2, 4)    