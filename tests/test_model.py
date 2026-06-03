from dl_assignment_2.modeling.model import SimpleRNN

from torch import Tensor, ones
class Test_model:
    rnn = SimpleRNN(10,20,4)
    def test_correct_output_shape(self):
        x: Tensor = ones((3, 10))   # BATCH, SEQ LEN, INPUT DIM
        
        y = self.rnn(x)
        assert y.shape == (1,4)
        
    def test_batch_correct_output_shape(self):
        x: Tensor = ones(5,3, 10)
        y = self.rnn(x)
        
        assert y.shape == (1, 5, 4) # always adds extra dimension at axis=0